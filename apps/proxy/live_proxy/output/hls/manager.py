"""
HLS Output Manager

Reads from the shared TS Redis buffer, splits the stream into
keyframe-aligned TS segments (pure packet copy, no remux, no subprocess;
see segmenter.py), stores one segment per Redis chunk via the shared
format-parameterized output buffer, and maintains a rolling live playlist
descriptor in Redis that the playlist view renders per request.

One instance per channel per cluster - coordinated via the shared
output:{fmt}:owner lock, exactly like the fMP4 remux manager.
"""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

from core.utils import RedisClient
from ..fmp4.buffer import FMP4StreamBuffer
from .segmenter import Part, TSSegmenter, client_is_stale, client_stale_after
from ...redis_keys import RedisKeys
from ...config_helper import ConfigHelper
from ...utils import get_logger

logger = get_logger()

# Output manager states stored in Redis (shared vocabulary with fMP4)
HLS_STATE_INITIALIZING = "initializing"
HLS_STATE_ACTIVE = "active"
HLS_STATE_STOPPED = "stopped"

# Redis TTL for state/owner/playlist keys
HLS_KEY_TTL = 3600

# Defaults; both overridable via proxy settings
DEFAULT_SEGMENT_DURATION = 4
# Retain 10 segments (~40s) in the rolling live window. A player starts
# near the live edge regardless of window length, so a longer window adds
# no latency; it only keeps older segments available so a client that
# briefly falls behind (a stall, a slow network hiccup) can still fetch the
# segment it is on instead of getting a 404 once it has rolled off.
DEFAULT_WINDOW_SIZE = 10
# Slack allowed above the advertised EXT-X-TARGETDURATION before a segment
# is force-cut. RFC 8216 4.3.3.1 requires each EXTINF to be <= the target
# WHEN ROUNDED TO THE NEAREST INTEGER, so a 4.49s segment is legal against a
# target of 4 but a 4.5s one is not.
#
# Not 0.49, even though that is where rounding actually flips. A cut lands on
# the first picture at or past the ceiling, so the emitted EXTINF overshoots
# it by up to one frame interval, and a ceiling of 4.49 emitted a 4.538s
# segment against a target of 4 - which rounds to 5 and contradicts the
# frozen value. The margin leaves room for that overshoot on low frame rates.
#
# Why this is as tight as it is: TARGETDURATION is not merely a ceiling, it
# is the client's playlist reload interval (RFC 8216 6.3.4). Advertising
# more than one segment's worth makes the player reload more slowly than
# segments are produced, so it loses buffer lead every cycle. Measured with
# AVPlayer against 4s segments advertised as 6: the lead eroded from 7s to
# 1s in twelve seconds and playback stalled with four seconds of media
# sitting unfetched on the server. Advertising the cut target itself keeps
# reload cadence and production cadence equal.
TARGET_ROUNDING_SLACK = 0.35

# Demand self-check. HLS clients are pull-based: there is no long-lived
# response whose teardown reports the disconnect, so the manager itself
# periodically verifies that at least one client is still FETCHING this
# output, and retires through the server's shared demand accounting when
# none has for two consecutive checks.
#
# The interval is part of how long an upstream connection - and the provider
# slot behind it - outlives the player that wanted it. Total release time is
# roughly the staleness window (see client_stale_after) plus up to
# GRACE x INTERVAL, so a slow poll here directly delays freeing a slot for
# the next tune-in. Cheap to run: one pipelined read of the client set.
DEMAND_CHECK_INTERVAL = 5
DEMAND_GRACE_CHECKS = 2

# Low-Latency HLS partial-segment target (seconds); 0 disables LL-HLS
# (segments only). Set HLS_PART_TARGET to ~0.5 to opt in, which puts the live
# edge within ~1.5s (PART-HOLD-BACK = 3 x PART-TARGET) for players that support
# Blocking Playlist Reload.
#
# OFF by default, because enabling it is not a transparent addition: an LL
# playlist advertises EXT-X-VERSION:10, and a client that does not implement
# version 10 is required to refuse the playlist outright rather than fall back
# to the whole segments it also carries. The whole-segment path is the one
# every client shares, so it stays the default and LL is opted into per
# deployment once its players are known to handle it.
DEFAULT_PART_TARGET = 0.0
# A part must stay fetchable while it is advertised (up to ~3 segments back)
# AND long enough for a blocking request already in flight to be answered. Sized
# generously: it costs only Redis memory, while too short a TTL expires an
# advertised part and 404s a client that asks for it.
PART_KEY_TTL = 60
# How many recent segments keep their parts in the descriptor and get their
# EXT-X-PART lines rendered. Matches Apple's ~3-target-durations guidance and
# the renderer's PARTS_RENDERED_SEGMENTS.
PARTS_RETAINED_SEGMENTS = 3


class HLSOutputManager:
    """
    Reads the TS Redis buffer for a channel, cuts keyframe-aligned HLS
    segments, and publishes them plus a rolling playlist window to Redis.
    """

    def __init__(self, channel_id, ts_buffer, worker_id, fmt='hls'):
        self.channel_id = channel_id
        self.ts_buffer = ts_buffer
        self.worker_id = worker_id
        self.fmt = fmt
        self.running = False
        self._thread = None
        # Set by the input side (StreamManager.update_url) when the upstream
        # switched; the next emitted segment is marked as a discontinuity.
        self._switch_pending = False

        self.segment_duration = ConfigHelper.get('HLS_SEGMENT_DURATION', DEFAULT_SEGMENT_DURATION)
        self.window_size = ConfigHelper.get('HLS_WINDOW_SIZE', DEFAULT_WINDOW_SIZE)
        # Advertised EXT-X-TARGETDURATION, computed ONCE and frozen for the life
        # of the playlist (RFC 8216 6.2.1: it MUST NOT change across reloads;
        # AVPlayer latches it at first parse and revalidates every reload).
        # Equal to the cut target: the segmenter aims each cut at the largest
        # whole number of GOPs fitting at or under that target, so segments
        # round to it, and matching the two keeps the client's reload cadence
        # equal to segment production (see TARGET_ROUNDING_SLACK).
        self.adv_target = max(int(round(self.segment_duration)), 1)
        # Force-cut ceiling handed to the segmenter. A segment may only run
        # past the advertised target by the rounding slack; beyond that it
        # would round up and contradict the frozen value.
        self.max_segment_duration = self.adv_target + TARGET_ROUNDING_SLACK
        # LL-HLS emit threshold, and the FROZEN advertised PART-TARGET derived
        # from it. adv_part sits just above the emit threshold so every part is
        # <= it while non-final parts stay >= 85% of it (RFC 8216bis 4.4.4.9);
        # like TARGETDURATION it is a stream-lifetime constant, since
        # rfc8216bis 6.2.1's permitted-change list excludes EXT-X-PART-INF and
        # EXT-X-SERVER-CONTROL.
        # Coerced: an unparseable value here would otherwise raise inside the
        # segmenter thread and take the whole manager down silently, where
        # falling back to the default just means "no LL" at worst.
        try:
            self.part_target = float(
                ConfigHelper.get('HLS_PART_TARGET', DEFAULT_PART_TARGET))
        except (TypeError, ValueError):
            self.part_target = DEFAULT_PART_TARGET
        self.adv_part = round(self.part_target * 1.12, 3) if self.part_target > 0 else 0.0

        # Same Redis-backed chunk store the fMP4 manager uses; it is
        # format-parameterized by design ("adding a new output format only
        # requires a new manager" - redis_keys.py). One HLS segment per
        # chunk; the chunk index doubles as the HLS media sequence number.
        self.segment_buffer = FMP4StreamBuffer(
            channel_id, redis_client=RedisClient.get_buffer(), fmt=fmt
        )
        # Size the chunk TTL to the advertised window plus ~one playlist of
        # post-removal availability (RFC 8216 6.2.2): a listed segment must stay
        # fetchable while in the playlist and for about a playlist duration after
        # it rolls off. A short default TTL cannot back a 10-segment window of
        # 5-6.5s segments, which 404s the window tail during stall recovery.
        try:
            self.segment_buffer.chunk_ttl = max(
                self.segment_buffer.chunk_ttl,
                int(self.window_size * (self.segment_duration + 3) + 30),
            )
        except Exception:
            pass
        self._redis = RedisClient.get_client()
        self._window = []
        # Video codec family ("h264"/"h265"/...) learned from the PMT once the
        # segmenter has parsed it; surfaced in the playlist descriptor so the
        # playlist view can refuse formats a client cannot decode (HEVC-in-TS).
        self._video_codec = None
        # Low-Latency HLS part state. _building_seq is the media sequence the
        # in-progress segment will take when it closes (put_fragment INCRs, so
        # it is the current index + 1); _building_parts accumulates
        # [dur, independent] for that segment; _parts_by_seq keeps a small tail
        # of completed segments' parts for the descriptor. _building_disc is
        # refreshed from the segmenter each loop iteration so building["disc"]
        # is truthful from the first published part.
        self._building_seq = None
        self._building_parts = []
        self._building_disc = False
        self._parts_by_seq = {}
        # Seed the rolling window + frozen target from an existing descriptor so
        # a mid-session worker restart/takeover does not clobber the playlist to
        # a fresh window (MEDIA-SEQUENCE must never regress; RFC 8216 6.2.2). The
        # FMP4StreamBuffer already restores its chunk index from Redis, so the
        # seeded window's seqs line up with the segments still in the buffer.
        if self._redis:
            try:
                existing = self._redis.get(RedisKeys.output_playlist(self.channel_id, self.fmt))
                if existing:
                    prior = json.loads(existing)
                    if prior.get("window"):
                        self._window = prior["window"]
                    if prior.get("adv_target"):
                        # Inherit the frozen value AND the ceiling derived from
                        # it, so a takeover cannot start emitting segments that
                        # round above the target the playlist already advertises.
                        self.adv_target = prior["adv_target"]
                        self.max_segment_duration = (
                            self.adv_target + TARGET_ROUNDING_SLACK
                        )
                    if prior.get("part_target"):
                        # Inherit the frozen PART-TARGET for the same reason:
                        # a takeover must not re-advertise a different one.
                        self.adv_part = prior["part_target"]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API (same surface as FMP4RemuxManager)
    # ------------------------------------------------------------------

    def start(self):
        """Acquire the output owner lock and spawn the segmenter thread."""
        if not self._acquire_owner_lock():
            logger.info(f"[HLS:{self.channel_id}] Another worker owns HLS output, skipping start")
            return False

        self.running = True
        self._set_state(HLS_STATE_INITIALIZING)

        short_id = self.channel_id[:8]
        self._thread = threading.Thread(
            target=self._segmenter_loop, daemon=True,
            name=f"hls-seg-{short_id}"
        )
        self._thread.start()

        logger.info(
            f"[HLS:{self.channel_id}] Started "
            f"(target={self.segment_duration}s, window={self.window_size})"
        )
        return True

    def stop(self):
        """Stop the segmenter thread and clean up all Redis keys."""
        if not self.running:
            return
        self.running = False
        logger.info(f"[HLS:{self.channel_id}] Stopping")

        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2)
            except Exception:
                pass

        self._cleanup_redis()
        logger.info(f"[HLS:{self.channel_id}] Stopped")

    def notify_stream_switch(self):
        """Input-side signal: the upstream stream changed (manual switch or
        automatic failover). The next emitted segment must carry
        EXT-X-DISCONTINUITY (RFC 8216 4.3.2.3)."""
        self._switch_pending = True

    # ------------------------------------------------------------------
    # Segmenter loop
    # ------------------------------------------------------------------

    def _segmenter_loop(self):
        """Read TS chunks from Redis and feed them through the segmenter."""
        segmenter = TSSegmenter(
            target_duration=self.segment_duration,
            max_segment_duration=self.max_segment_duration,
            part_target=self.part_target,
            # Clamp emitted parts to exactly the advertised PART-TARGET so no
            # EXT-X-PART DURATION can exceed the frozen constant.
            part_ceiling=self.adv_part,
        )

        # Start behind live so the first segments cover the same window a
        # new TS client would receive, matching fMP4 writer positioning.
        behind_seconds = ConfigHelper.new_client_behind_seconds()
        start_index = self.ts_buffer.find_chunk_index_by_time(behind_seconds) if behind_seconds > 0 else None
        if start_index is None:
            start_index = self.ts_buffer.index
        local_index = start_index
        first_segment_stored = False
        last_demand_check = time.time()
        idle_demand_checks = 0
        logger.debug(
            f"[HLS:{self.channel_id}] Segmenter started at buffer index "
            f"{local_index} ({behind_seconds}s behind live)"
        )

        try:
            while self.running:
                if self._switch_pending:
                    self._switch_pending = False
                    # Hard cut: close the open segment from pre-switch bytes
                    # only; the next segment starts at a post-switch keyframe
                    # and carries the discontinuity tag.
                    self._store_discontinuity_tail(segmenter.flag_discontinuity())
                    logger.info(
                        f"[HLS:{self.channel_id}] Input stream switched; segment "
                        f"cut, next segment will be marked as a discontinuity"
                    )

                now = time.time()
                if now - last_demand_check >= DEMAND_CHECK_INTERVAL:
                    last_demand_check = now
                    if self._has_hls_demand():
                        idle_demand_checks = 0
                    else:
                        idle_demand_checks += 1
                        if idle_demand_checks >= DEMAND_GRACE_CHECKS:
                            logger.info(
                                f"[HLS:{self.channel_id}] No {self.fmt} clients for "
                                f"{idle_demand_checks * DEMAND_CHECK_INTERVAL}s; retiring output"
                            )
                            self._retire()
                            if not self.running:
                                break
                            # A client tuned in while we were deciding: the
                            # server's authoritative accounting saw it and
                            # kept this manager alive, so keep segmenting
                            # for the newcomer instead of exiting and
                            # leaving a registered manager with a dead loop.
                            logger.info(
                                f"[HLS:{self.channel_id}] New {self.fmt} client "
                                f"arrived during retirement; resuming"
                            )
                            idle_demand_checks = 0

                chunks, new_index = self.ts_buffer.get_optimized_client_data(local_index)

                if chunks:
                    local_index = new_index
                    for chunk in chunks:
                        if not self.running:
                            break
                        events = segmenter.feed(chunk)
                        for i, event in enumerate(events):
                            self._video_codec = segmenter.video_codec
                            self._building_disc = segmenter.current_discontinuity
                            if isinstance(event, Part):
                                # Suppress the descriptor publish for a final
                                # part immediately followed by its Segment (the
                                # segment publish supersedes it microseconds
                                # later), so the transient state never
                                # advertises a PRELOAD-HINT for a part of a
                                # segment that is closing.
                                publish = not (
                                    i + 1 < len(events)
                                    and not isinstance(events[i + 1], Part)
                                )
                                self._store_part(event, publish=publish)
                                continue
                            segment = event
                            self._store_segment(segment)
                            if not first_segment_stored:
                                first_segment_stored = True
                                self._set_state(HLS_STATE_ACTIVE)
                                logger.info(
                                    f"[HLS:{self.channel_id}] First segment stored "
                                    f"({segment.duration:.2f}s, {len(segment.data)} bytes)"
                                )
                else:
                    if self.ts_buffer.index > local_index + 20:
                        # Fell too far behind (slow consumer / provider burst):
                        # skip forward and mark the gap for the playlist. The
                        # open segment is hard-cut so pre-gap and post-gap
                        # data never share a segment.
                        local_index = self.ts_buffer.index - 5
                        self._store_discontinuity_tail(segmenter.flag_discontinuity())
                        logger.debug(
                            f"[HLS:{self.channel_id}] Skipped forward to index {local_index}"
                        )
                    time.sleep(0.05)

        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Segmenter loop error: {e}", exc_info=True)
        finally:
            logger.debug(f"[HLS:{self.channel_id}] Segmenter loop exited")

    def _store_discontinuity_tail(self, events):
        """Store whatever flag_discontinuity closed out: in LL mode the closing
        segment's final Part followed by the Segment itself, otherwise just the
        Segment. Empty when the open segment held nothing playable."""
        for event in events:
            if isinstance(event, Part):
                # Do not publish on this part: the Segment right behind it
                # supersedes the descriptor microseconds later.
                self._store_part(event, publish=False)
            else:
                self._store_segment(event)

    def _store_part(self, part, publish=True):
        """Store one Low-Latency HLS partial segment for the in-progress segment
        and refresh the descriptor so the live edge advances every ~part_target.
        ``publish=False`` stores the bytes but skips the descriptor refresh (used
        for a final part whose closing Segment publishes right after)."""
        if self.part_target <= 0:
            return
        if self._building_seq is None:
            # The in-progress segment takes the next media sequence number
            # (put_fragment INCRs the index when it is eventually stored).
            self._building_seq = self.segment_buffer.index + 1
        part_index = len(self._building_parts)
        # Store parts in the same buffer Redis as the segment chunks so the part
        # view reads them exactly like hls_segment reads chunks. A missing buffer
        # or a failed write must NOT append to _building_parts: the descriptor
        # would then advertise a part whose bytes were never stored, and every
        # request for that URI would 404 after a blocking hold.
        buf = self.segment_buffer.redis_client
        if not buf:
            return
        try:
            buf.setex(
                RedisKeys.output_part(
                    self.channel_id, self.fmt, self._building_seq, part_index
                ),
                PART_KEY_TTL,
                part.data,
            )
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error storing part: {e}")
            return
        self._building_parts.append([round(part.duration, 5), bool(part.independent)])
        # Publish only once a full segment anchors the window: a descriptor with
        # an empty window renders a degenerate zero-segment playlist AVPlayer
        # will not start on (the cold-start black screen). The bytes are still
        # stored above, ready the moment the first segment closes.
        if publish and self._window:
            self._publish_playlist_state()

    def _store_segment(self, segment):
        """Store one finished segment and refresh the playlist descriptor."""
        if not self.segment_buffer.put_fragment(segment.data):
            # Redis write failed: drop the in-progress LL state so the next part
            # re-derives its seq from Redis rather than accumulating two
            # segments' parts under a stale seq (which would 404 every
            # advertised part).
            self._building_parts = []
            self._building_seq = None
            self._building_disc = False
            return
        seq = self.segment_buffer.index
        # Wall-clock anchor for the segment START (Apple's Low-Latency profile
        # requires EXT-X-PROGRAM-DATE-TIME on all media playlists; it also
        # drives AVPlayer's recommendedTimeOffsetFromLive).
        seg_start = datetime.now(timezone.utc) - timedelta(seconds=segment.duration)
        self._window.append({
            "seq": seq,
            "dur": round(segment.duration, 3),
            "disc": bool(segment.discontinuity),
            "pdt": seg_start.isoformat(timespec="milliseconds"),
        })
        if len(self._window) > self.window_size:
            self._window = self._window[-self.window_size:]

        # The parts accumulated while building now belong to this completed
        # segment (its seq equals the seq tracked during building). Hand them
        # over, start a fresh in-progress segment, and prune old parts.
        if self.part_target > 0:
            self._parts_by_seq[str(seq)] = self._building_parts
            self._building_parts = []
            self._building_seq = self.segment_buffer.index + 1
            keep = {str(e["seq"]) for e in self._window[-PARTS_RETAINED_SEGMENTS:]}
            self._parts_by_seq = {
                k: v for k, v in self._parts_by_seq.items() if k in keep
            }

        self._publish_playlist_state()

        # Heartbeat the owner lock and state key (both set once with ex=3600 and
        # otherwise never refreshed): a stream longer than an hour would silently
        # lose mutual exclusion and let a second worker start a duplicate
        # segmenter, breaking MEDIA-SEQUENCE monotonicity. If ownership has moved,
        # stop cleanly rather than fight the new owner.
        self._heartbeat_ownership()

        logger.debug(
            f"[HLS:{self.channel_id}] Segment {seq}: "
            f"{segment.duration:.2f}s, {len(segment.data)} bytes"
            f"{' [discontinuity]' if segment.discontinuity else ''}"
        )

    def _publish_playlist_state(self):
        """Write the rolling playlist descriptor to Redis for the playlist view
        to render on demand. Includes LL-HLS part data when enabled."""
        if not self._redis:
            return
        try:
            playlist_state = {
                "window": self._window,
                "target": self.segment_duration,
                "adv_target": self.adv_target,
                "vcodec": self._video_codec,
            }
            if self.part_target > 0:
                # Carries the FROZEN advertised PART-TARGET (adv_part), not the
                # raw emit threshold, so PART-TARGET and PART-HOLD-BACK are
                # constant across reloads (rfc8216bis 6.2.1).
                playlist_state["part_target"] = self.adv_part
                playlist_state["parts"] = self._parts_by_seq
                playlist_state["building"] = {
                    "seq": self._building_seq,
                    "parts": self._building_parts,
                    "disc": self._building_disc,
                }
            self._redis.setex(
                RedisKeys.output_playlist(self.channel_id, self.fmt),
                HLS_KEY_TTL,
                json.dumps(playlist_state),
            )
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error updating playlist state: {e}")

    # ------------------------------------------------------------------
    # Demand accounting (pull-based clients)
    # ------------------------------------------------------------------

    def _has_hls_demand(self):
        """True when at least one client is still FETCHING this manager's
        output. Mirrors the per-format accounting in handle_client_disconnect;
        set entries whose metadata hash has expired are ghosts and do not
        count as demand.

        Existence of the record is not enough. A pull-based client cannot
        report a disconnect, so its record simply sits at whatever TTL it was
        last given (CLIENT_RECORD_TTL, a minute) after the player has stopped
        polling - and for that whole minute the upstream connection, and the
        provider slot behind it, stays claimed. A streaming client releases
        its slot the moment its response tears down; this closes most of that
        gap by judging demand on when a client last actually fetched, and by
        releasing the records that fail that test instead of waiting for them
        to lapse on their own.
        """
        if not self._redis:
            return True  # cannot verify; err on the side of running
        try:
            client_ids = list(self._redis.smembers(RedisKeys.clients(self.channel_id)))
            if not client_ids:
                return False
            pipe = self._redis.pipeline(transaction=False)
            for cid in client_ids:
                key = RedisKeys.client_metadata(self.channel_id, cid)
                pipe.hget(key, "output_format")
                pipe.hget(key, "output_profile_id")
                pipe.hget(key, "last_active")
            results = pipe.execute()

            now = time.time()
            stale_after = client_stale_after(self.segment_duration)
            demand = False
            stale_ids = []
            for idx, cid in enumerate(client_ids):
                fmt = results[idx * 3]
                if not fmt:
                    continue  # expired hash: a ghost entry, not demand
                fmt = fmt.decode() if isinstance(fmt, bytes) else fmt
                pid = results[idx * 3 + 1]
                pid = (pid.decode() if isinstance(pid, bytes) else pid) if pid else ''
                manager_key = fmt
                if pid:
                    try:
                        manager_key = f"{fmt}:p{int(pid)}"
                    except ValueError:
                        pass
                if manager_key != self.fmt:
                    continue
                if client_is_stale(results[idx * 3 + 2], now, stale_after):
                    stale_ids.append(cid)
                else:
                    demand = True

            if stale_ids:
                self._release_stale_clients(stale_ids, stale_after)
            return demand
        except Exception as e:
            logger.debug(f"[HLS:{self.channel_id}] Demand check failed: {e}")
            return True

    def _release_stale_clients(self, client_ids, stale_after):
        """Drop clients that stopped fetching, as a disconnect would.

        Deleting the metadata hash is the part that matters: the channel's
        shared accounting counts a client while its hash exists, so leaving
        the record to expire on its own is exactly what held the upstream
        open for a minute after the player went away.
        """
        try:
            from ...server import ProxyServer
            mgr = ProxyServer.get_instance().client_managers.get(self.channel_id)
        except Exception:
            mgr = None
        for cid in client_ids:
            cid_str = cid.decode() if isinstance(cid, bytes) else cid
            try:
                if mgr:
                    # Preferred: goes through the same bookkeeping (and the
                    # same last-client-disconnect trigger) a streaming
                    # client's teardown uses.
                    mgr.remove_client(cid_str)
                    continue
                # Owner runs on another worker: mirror what remove_client
                # does to Redis, matching _drop_pre_registered_client.
                self._redis.srem(RedisKeys.clients(self.channel_id), cid_str)
                self._redis.delete(
                    RedisKeys.client_metadata(self.channel_id, cid_str)
                )
            except Exception as e:
                logger.debug(
                    f"[HLS:{self.channel_id}] Could not release stale client "
                    f"{cid_str}: {e}"
                )
        logger.info(
            f"[HLS:{self.channel_id}] Released {len(client_ids)} client(s) with no "
            f"fetch in {stale_after:.0f}s; upstream slot freed without waiting "
            f"out the client record TTL"
        )

    def _retire(self):
        """No consumers remain: prune expired client-set entries, then hand
        teardown to the server's shared demand accounting so this manager is
        stopped AND deregistered (and the channel shuts down when nothing
        else remains), exactly as a streaming client's disconnect would.

        The server is the arbiter: if a client tuned in between our demand
        check and this call, the accounting keeps the manager registered and
        does NOT stop it; the caller must then keep the loop running
        (self.running stays True) rather than exit."""
        try:
            from ...client_manager import ClientManager
            ClientManager.remove_ghost_clients(self._redis, self.channel_id)
        except Exception:
            pass
        try:
            # Import locally: server imports this module at load time. This
            # runs in the manager's own thread; stop() tolerates the resulting
            # self-join (the RuntimeError is caught) and the loop exits right
            # after this call returns.
            from ...server import ProxyServer
            ProxyServer.get_instance().handle_client_disconnect(self.channel_id)
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error during retirement: {e}")

    # ------------------------------------------------------------------
    # Redis helpers (mirror FMP4RemuxManager)
    # ------------------------------------------------------------------

    def _acquire_owner_lock(self) -> bool:
        if not self._redis:
            return True
        owner_key = RedisKeys.output_owner(self.channel_id, self.fmt)
        acquired = self._redis.set(owner_key, self.worker_id, nx=True, ex=HLS_KEY_TTL)
        if acquired:
            return True
        existing = self._redis.get(owner_key)
        return existing == self.worker_id

    def _set_state(self, state: str):
        if self._redis:
            self._redis.setex(RedisKeys.output_state(self.channel_id, self.fmt), HLS_KEY_TTL, state)

    def _heartbeat_ownership(self):
        """Re-extend the owner lock + state TTL while we still own them; stop the
        loop if another worker has taken over. Called once per stored segment."""
        if not self._redis:
            return
        try:
            owner_key = RedisKeys.output_owner(self.channel_id, self.fmt)
            if self._redis.get(owner_key) == self.worker_id:
                self._redis.expire(owner_key, HLS_KEY_TTL)
                self._redis.expire(RedisKeys.output_state(self.channel_id, self.fmt), HLS_KEY_TTL)
            else:
                logger.info(f"[HLS:{self.channel_id}] Output ownership moved to another worker; stopping")
                self.running = False
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Ownership heartbeat error: {e}")

    def _cleanup_redis(self):
        """Delete all HLS output Redis keys for this channel."""
        if not self._redis:
            return
        try:
            keys_to_delete = [
                RedisKeys.output_state(self.channel_id, self.fmt),
                RedisKeys.output_owner(self.channel_id, self.fmt),
                RedisKeys.output_playlist(self.channel_id, self.fmt),
            ]
            self._redis.delete(*keys_to_delete)
            self.segment_buffer.cleanup_redis()
        except Exception as e:
            logger.error(f"[HLS:{self.channel_id}] Error during Redis cleanup: {e}")
