"""
MPEG-TS HLS segmenter - pure packet-copy splitting, no remux.

The live proxy's source ring already guarantees 188-byte packet alignment
(StreamBuffer.add_chunk), and TS segments are first-class HLS citizens
(RFC 8216 section 3.2), so producing HLS from the ring is a matter of
CUTTING the existing packets into keyframe-aligned segments. No bytes are
rewritten, no subprocess is spawned.

This module is intentionally dependency-free (stdlib only, no Django or
Redis imports) so the parsing logic is unit-testable in isolation.

Segmentation rules:
- A segment may only begin on a video keyframe access unit. Keyframes are
  detected via the adaptation-field random_access_indicator when the
  provider sets it, with a fallback NAL-header scan (H.264 IDR/SPS,
  H.265 IRAP/parameter sets) for providers that do not.
- Segment duration is measured from video PES PTS deltas, cut at the
  first keyframe at or after the target duration.
- Every emitted segment is prefixed with the most recently seen PAT and
  PMT packets so each segment decodes independently, as HLS requires.
"""

TS_PACKET_SIZE = 188
TS_SYNC_BYTE = 0x47

# ISO 13818-1 / ATSC stream_type values
VIDEO_STREAM_TYPES = {
    0x01: "mpeg1",
    0x02: "mpeg2",
    0x1B: "h264",
    0x24: "h265",
}

PTS_CLOCK = 90000.0
# 33-bit PTS wraps every ~26.5 hours; treat large negative deltas as a wrap.
PTS_WRAP = 1 << 33


class Segment:
    """One finished HLS media segment."""

    __slots__ = ("data", "duration", "discontinuity")

    def __init__(self, data, duration, discontinuity=False):
        self.data = data
        self.duration = duration
        self.discontinuity = discontinuity


def packet_pid(packet):
    """13-bit PID of a TS packet."""
    return ((packet[1] & 0x1F) << 8) | packet[2]


def packet_pusi(packet):
    """payload_unit_start_indicator flag."""
    return bool(packet[1] & 0x40)


def packet_payload_offset(packet):
    """Byte offset of the payload within the packet, or None if no payload."""
    afc = (packet[3] >> 4) & 0x03
    if afc == 0x01:
        return 4
    if afc == 0x03:
        af_len = packet[4]
        offset = 5 + af_len
        return offset if offset < TS_PACKET_SIZE else None
    return None


def packet_random_access(packet):
    """adaptation-field random_access_indicator, when an AF is present."""
    afc = (packet[3] >> 4) & 0x03
    if afc in (0x02, 0x03) and packet[4] > 0:
        return bool(packet[5] & 0x40)
    return False


def parse_pat(packet):
    """Return the PMT PID of the first non-zero program, or None."""
    base = packet_payload_offset(packet)
    if base is None or base + 1 >= TS_PACKET_SIZE:
        return None
    pointer = packet[base]
    section = base + 1 + pointer
    # table_id(1) section_length(2) tsid(2) ver(1) sec(1) last(1) = 8 bytes,
    # then 4-byte program entries.
    offset = section + 8
    while offset + 3 < TS_PACKET_SIZE:
        program_number = (packet[offset] << 8) | packet[offset + 1]
        pid = ((packet[offset + 2] & 0x1F) << 8) | packet[offset + 3]
        if program_number != 0:
            return pid
        offset += 4
    return None


def parse_pmt(packet):
    """Return (video_pid, video_stream_type) from a PMT packet, or (None, None)."""
    base = packet_payload_offset(packet)
    if base is None or base + 1 >= TS_PACKET_SIZE:
        return None, None
    pointer = packet[base]
    section = base + 1 + pointer
    if section + 12 >= TS_PACKET_SIZE:
        return None, None
    section_length = ((packet[section + 1] & 0x0F) << 8) | packet[section + 2]
    program_info_length = ((packet[section + 10] & 0x0F) << 8) | packet[section + 11]
    offset = section + 12 + program_info_length
    section_end = min(section + 3 + section_length - 4, TS_PACKET_SIZE - 1)

    while offset + 4 < section_end:
        stream_type = packet[offset]
        es_pid = ((packet[offset + 1] & 0x1F) << 8) | packet[offset + 2]
        es_info_length = ((packet[offset + 3] & 0x0F) << 8) | packet[offset + 4]
        if stream_type in VIDEO_STREAM_TYPES:
            return es_pid, stream_type
        offset += 5 + es_info_length
    return None, None


def extract_pts(packet):
    """PTS in seconds from a PES header starting in this packet, or None."""
    base = packet_payload_offset(packet)
    if base is None or base + 13 >= TS_PACKET_SIZE:
        return None
    if packet[base] != 0x00 or packet[base + 1] != 0x00 or packet[base + 2] != 0x01:
        return None
    flags = packet[base + 7]
    if not (flags & 0x80):
        return None
    b = packet
    pts = (
        ((b[base + 9] >> 1) & 0x07) << 30
        | b[base + 10] << 22
        | ((b[base + 11] >> 1) & 0x7F) << 15
        | b[base + 12] << 7
        | (b[base + 13] >> 1)
    )
    return pts / PTS_CLOCK


def starts_keyframe(packet, video_stream_type):
    """
    Does this PUSI video packet open a keyframe access unit?

    Prefers the adaptation-field random_access_indicator; falls back to
    scanning visible NAL start codes. Encoders emit parameter sets
    immediately before IDR/IRAP frames, so SPS/VPS in the first packet is
    a reliable keyframe marker even when the keyframe NAL itself starts
    in a later packet of the same PES.
    """
    if packet_random_access(packet):
        return True

    base = packet_payload_offset(packet)
    if base is None or base + 9 >= TS_PACKET_SIZE:
        return False
    header_len = packet[base + 8]
    i = base + 9 + header_len
    end = TS_PACKET_SIZE - 4
    while i < end:
        if packet[i] == 0x00 and packet[i + 1] == 0x00:
            nal_start = -1
            if packet[i + 2] == 0x01:
                nal_start = i + 3
            elif packet[i + 2] == 0x00 and i + 3 < end and packet[i + 3] == 0x01:
                nal_start = i + 4
            if 0 < nal_start < TS_PACKET_SIZE:
                if video_stream_type == 0x24:
                    # H.265: nal_unit_type in bits 1-6 of the first byte.
                    nal_type = (packet[nal_start] >> 1) & 0x3F
                    # IRAP (16-21) or VPS/SPS/PPS (32-34)
                    if 16 <= nal_type <= 21 or 32 <= nal_type <= 34:
                        return True
                else:
                    # H.264: nal_unit_type in bits 0-4.
                    nal_type = packet[nal_start] & 0x1F
                    # IDR (5) or SPS (7)
                    if nal_type in (5, 7):
                        return True
                i = nal_start
                continue
        i += 1
    return False


class TSSegmenter:
    """
    Stateful packet-copy segmenter. Feed it raw TS bytes (any chunking);
    it returns finished Segment objects as keyframe boundaries are crossed.
    """

    def __init__(self, target_duration=4.0, max_segment_duration=None,
                 startup_keyframe_cuts=4, startup_ramp_fractions=(0.5, 0.75),
                 startup_min_duration=1.0):
        self.target_duration = float(target_duration)
        # Hard ceiling: force a cut before a segment can exceed this, so no
        # emitted EXTINF ever exceeds the frozen advertised TARGETDURATION even
        # on a keyframe drought (RFC 8216 4.3.3.1). Defaults to 2x the target.
        self.max_segment_duration = float(
            max_segment_duration if max_segment_duration else 2 * target_duration)
        # Fast-start ladder: a cold channel accumulates segments at live
        # cadence, so with a 4s target a player waits ~8-12s for enough
        # media to start. The first N segments therefore cut at EVERY
        # keyframe (one GOP each, typically 1-3s), which gets a playable
        # playlist up in one GOP and 3 segments within a few seconds.
        self._startup_cuts_remaining = int(startup_keyframe_cuts)
        # ...and then the cut target RAMPS back to normal rather than
        # stepping there in one go. The starter segments are cut from the
        # backlog the segmenter starts behind live, so they are produced far
        # faster than 1x; the first full-length segment after them is the
        # first one produced at live cadence. Jumping straight from a 1-GOP
        # starter to a full target means the player's next segment can be a
        # whole target-duration away exactly when its starter buffer runs
        # dry, which stalls it a few seconds into playback. Intermediate
        # steps keep the grain fine across that handover, so the wait for
        # the next segment grows gradually instead of doubling at once.
        # Steady-state output is unchanged, and every ramp EXTINF stays
        # under the frozen TARGETDURATION.
        self._startup_ramp = [
            float(f) * self.target_duration for f in (startup_ramp_fractions or ())
            if 0 < float(f) < 1
        ]
        # Floor on starter segment length. Cutting at EVERY keyframe means the
        # ladder acts on every keyframe the detector reports - including false
        # positives from the NAL fallback scan, which a steady-state target
        # threshold silently absorbs. Observed on a live source: starters of
        # 0.97s, 1.03s and 0.30s where the real GOP was 2.0s, so the four
        # starter slots carried 4.3s of runway instead of ~8s. The floor costs
        # nothing when keyframes are honest (a 2s GOP still cuts at 2s) and
        # stops a spurious keyframe from spending a starter slot on a fragment.
        self.startup_min_duration = float(startup_min_duration or 0)
        self._pending = bytearray()
        self._current = bytearray()
        self._pat_packet = None
        self._pmt_packet = None
        self._pmt_pid = None
        self._video_pid = None
        self._video_stream_type = None
        self._segment_start_pts = None
        # First / most-recent video PTS in the current segment; used to report a
        # MEASURED duration on the discontinuity cut instead of substituting the
        # nominal target (RFC 8216 4.3.2.1: EXTINF must be accurate).
        self._seg_first_pts = None
        self._seg_last_pts = None
        self._collecting = False
        self._pending_discontinuity = False
        self._current_discontinuity = False
        # Running estimate of the source's keyframe interval, measured from
        # the gap between consecutive keyframes. Needed because the cut rule
        # aims at the largest whole number of GOPs that still fits INSIDE the
        # target (see _cut_threshold); until a second keyframe has been seen
        # there is nothing to estimate from and the rule falls back to cutting
        # at the first keyframe past the target.
        self._gop_estimate = None
        self._last_keyframe_pts = None

    @property
    def video_detected(self):
        return self._video_pid is not None

    @property
    def video_codec(self):
        """Detected video codec family name (e.g. "h264", "h265"), or None
        until the PMT has been parsed. Used to advertise the codec to
        clients and to gate formats that a given HLS client cannot decode
        (notably HEVC-in-MPEG-TS, which AVFoundation refuses)."""
        return VIDEO_STREAM_TYPES.get(self._video_stream_type)

    def flag_discontinuity(self):
        """Mark a stream discontinuity (provider failover, buffer skip-ahead).

        Hard cut: the in-progress segment is closed IMMEDIATELY from the bytes
        already collected, so pre-gap and post-gap data can never share a
        segment. Collection resumes at the next keyframe, and that new segment
        is the one tagged with EXT-X-DISCONTINUITY.

        Returns the finished pre-gap Segment, or None when the open segment
        held nothing playable (its measured span is zero) and was discarded.
        """
        finished = None
        if self._collecting:
            span = 0.0
            if self._seg_first_pts is not None and self._seg_last_pts is not None:
                span = self._elapsed(self._seg_last_pts, self._seg_first_pts)
            if span > 0:
                finished = self._finish_segment(span)
        # Drop any un-finished remainder and wait for the next keyframe; the
        # PTS timeline may jump arbitrarily across the discontinuity.
        self._collecting = False
        self._current = bytearray()
        self._current_discontinuity = False
        self._segment_start_pts = None
        self._seg_first_pts = None
        self._seg_last_pts = None
        self._pending_discontinuity = True
        return finished

    def feed(self, data):
        """Consume raw TS bytes; return a list of finished Segments (possibly empty)."""
        segments = []
        self._pending.extend(data)

        while len(self._pending) >= TS_PACKET_SIZE:
            if self._pending[0] != TS_SYNC_BYTE:
                sync = self._pending.find(bytes([TS_SYNC_BYTE]))
                if sync < 0:
                    self._pending.clear()
                    break
                del self._pending[:sync]
                continue
            # Require the next packet to also be in sync (or be the tail) so
            # a stray 0x47 in payload cannot fake an alignment point.
            if (
                len(self._pending) >= TS_PACKET_SIZE + 1
                and self._pending[TS_PACKET_SIZE] != TS_SYNC_BYTE
            ):
                del self._pending[:1]
                continue

            packet = bytes(self._pending[:TS_PACKET_SIZE])
            del self._pending[:TS_PACKET_SIZE]
            finished = self._handle_packet(packet)
            if finished is not None:
                segments.append(finished)

        return segments

    def _handle_packet(self, packet):
        pid = packet_pid(packet)

        if pid == 0:
            self._pat_packet = packet
            if self._pmt_pid is None:
                self._pmt_pid = parse_pat(packet)
            return None
        if self._pmt_pid is not None and pid == self._pmt_pid:
            self._pmt_packet = packet
            video_pid, stream_type = parse_pmt(packet)
            if video_pid is not None:
                # Re-learned continuously so PID/codec changes across
                # provider failovers are tolerated.
                self._video_pid = video_pid
                self._video_stream_type = stream_type
            return None

        if self._video_pid is None:
            return None

        finished = None
        if pid == self._video_pid and packet_pusi(packet):
            pts = extract_pts(packet)
            keyframe = starts_keyframe(packet, self._video_stream_type)
            if pts is not None:
                self._seg_last_pts = pts
            if keyframe and pts is not None:
                self._note_keyframe(pts)

            if not self._collecting:
                if keyframe:
                    self._begin_segment(pts)
            elif keyframe and pts is not None:
                if self._segment_start_pts is None:
                    # Discontinuity reset the timeline: cut here, reporting the
                    # measured span of the segment being closed (RFC 8216 4.3.2.1)
                    # rather than substituting the nominal target.
                    finished = self._finish_segment(self._measured_span())
                    self._begin_segment(pts)
                else:
                    elapsed = self._elapsed(pts, self._segment_start_pts)
                    # Fast-start ladder, then ramp; elapsed > 0 skips
                    # same-PTS duplicates.
                    cut_at = self._cut_threshold()
                    if elapsed >= cut_at and elapsed > 0:
                        finished = self._finish_segment(elapsed)
                        self._begin_segment(pts)
            elif pts is not None and self._collecting and self._segment_start_pts is not None:
                # Keyframe drought: force a cut so the segment cannot exceed the
                # frozen TARGETDURATION. Cutting mid-GOP yields a segment that is
                # not keyframe-independent, an accepted last resort that a healthy
                # GOP (which cuts on its keyframes well under this ceiling) never
                # reaches.
                elapsed = self._elapsed(pts, self._segment_start_pts)
                if elapsed >= self.max_segment_duration:
                    finished = self._finish_segment(elapsed)
                    self._begin_segment(pts)

        if self._collecting:
            self._current.extend(packet)
        return finished

    def _cut_threshold(self):
        """Elapsed time at which the next keyframe may close the segment.

        The starter floor while starter cuts remain, then each ramp step in
        turn, then steady state.

        Steady state aims at the largest whole number of GOPs that fits at or
        under the target, rather than the first keyframe at or AFTER it. The
        difference decides what TARGETDURATION can be, and TARGETDURATION is
        the client's playlist reload interval (RFC 8216 6.3.4): advertise more
        than a segment's worth and the player reloads more slowly than
        segments are produced, losing buffer lead every cycle until it stalls.
        Measured with AVPlayer against a 4s target advertised as 6: the lead
        eroded from 7s to 1s in twelve seconds and the player ran dry with
        four seconds of media sitting unfetched on the server.

        Overshooting is what forced that headroom. A 3s GOP against a 4s
        target used to cut at the first keyframe past 4s - a 6s segment,
        needing TARGETDURATION 6. Subtracting one GOP from the target cuts at
        3s instead, so the segment lands under the target and the advertised
        value can equal it. Sources whose GOP is longer than the target cannot
        satisfy both constraints and are handled by the force-cut ceiling.
        """
        if self._startup_cuts_remaining > 0:
            return self.startup_min_duration
        if self._startup_ramp:
            return max(self._startup_ramp[0], self.startup_min_duration)
        if not self._gop_estimate:
            return self.target_duration
        # The epsilon keeps a keyframe landing exactly one GOP short of the
        # target from cutting there: with a 2s GOP and a 4s target the
        # threshold is 2.01, so the 2s keyframe is skipped and the 4s one
        # cuts - a full target-length segment, not a half-length one.
        return max(self.target_duration - self._gop_estimate + 0.01, 0.01)

    def _note_keyframe(self, pts):
        """Fold one keyframe interval into the running GOP estimate.

        Deliberately conservative: it tracks the LARGEST recent interval
        rather than the mean. The estimate is subtracted from the target, so
        underestimating it makes segments overshoot the target - the very
        thing that forced the oversized TARGETDURATION. A spurious keyframe
        reports a short interval, which must not drag the estimate down.
        """
        if self._last_keyframe_pts is not None:
            gap = self._elapsed(pts, self._last_keyframe_pts)
            # Ignore nonsense: duplicates, and jumps past a plausible GOP.
            if 0 < gap <= 4 * self.target_duration:
                if self._gop_estimate is None:
                    self._gop_estimate = gap
                else:
                    # Rise immediately, decay slowly.
                    self._gop_estimate = max(gap, self._gop_estimate * 0.9)
        self._last_keyframe_pts = pts

    def _elapsed(self, pts, start):
        """Wrap-safe presentation-time delta in seconds."""
        d = pts - start
        if d < 0:
            d += PTS_WRAP / PTS_CLOCK
        return d

    def _measured_span(self):
        """Best measured duration of the segment being closed, from the first and
        last video PTS seen. Falls back to the target only when unmeasurable or
        nonsensical (e.g. a timeline jump)."""
        if self._seg_first_pts is None or self._seg_last_pts is None:
            return self.target_duration
        d = self._elapsed(self._seg_last_pts, self._seg_first_pts)
        if d <= 0 or d > 4 * self.target_duration:
            return self.target_duration
        return d

    def _begin_segment(self, pts):
        self._current = bytearray()
        if self._pat_packet:
            self._current.extend(self._pat_packet)
        if self._pmt_packet:
            self._current.extend(self._pmt_packet)
        self._segment_start_pts = pts
        self._seg_first_pts = pts
        self._seg_last_pts = pts
        self._collecting = True
        self._current_discontinuity = self._pending_discontinuity
        self._pending_discontinuity = False

    def _finish_segment(self, duration):
        if duration <= 0 or duration > 4 * self.target_duration:
            duration = self.target_duration
        segment = Segment(
            bytes(self._current),
            float(duration),
            discontinuity=self._current_discontinuity,
        )
        self._current = bytearray()
        self._current_discontinuity = False
        if self._startup_cuts_remaining > 0:
            self._startup_cuts_remaining -= 1
        elif self._startup_ramp:
            self._startup_ramp.pop(0)
        return segment


# A player starts on whatever the FIRST playlist it reads contains, and on a
# cold channel that playlist exists as soon as one segment does. Handing it a
# one- or two-segment window means it begins on a couple of seconds of media
# cut from the pre-roll backlog, drains that at 1x, and starves at the handover
# to live-cadence production - a freeze a few seconds in, after which it
# re-buffers deeper and never stalls again. Holding the first response until
# the window can sustain playback costs about a second of tune-in and removes
# the stall. An established channel is already well past this threshold, so a
# mid-session reload never waits.
MIN_START_SEGMENTS = 3

# Where a joining player is told to start, as a multiple of the cut target,
# and therefore also how much media the window must hold before it is served.
# One constant for both: EXT-X-START promising a join point 10s behind the
# live edge while the window holds 6s is a promise the playlist cannot keep,
# and the player silently starts at the window head with correspondingly less
# runway than it was told to expect.
LIVE_EDGE_OFFSET_FACTOR = 2.5


# How many missed playlist fetches mark an HLS client as gone, and the floor
# under that in seconds.
#
# A pull-based client cannot report a disconnect, so its absence has to be
# inferred - but it can be inferred far faster than the generic
# CLIENT_RECORD_TTL allows. A player reloads the media playlist about once
# per TARGETDURATION (RFC 8216 6.3.4), and TARGETDURATION now equals the cut
# target, so three missed reloads is unambiguous. The floor keeps a short
# target from reaping a client over one slow network moment.
CLIENT_STALE_POLL_MULTIPLE = 3
CLIENT_STALE_FLOOR = 15.0


def client_stale_after(target_duration):
    """Seconds without a playlist or segment fetch after which an HLS client
    is treated as gone, so its upstream slot can be released."""
    try:
        target = float(target_duration)
    except (TypeError, ValueError):
        target = 4.0
    return max(CLIENT_STALE_POLL_MULTIPLE * target, CLIENT_STALE_FLOOR)


def client_is_stale(last_active, now, stale_after):
    """True when a client's last fetch is old enough to call it gone.

    An unreadable or missing timestamp is never stale: the record may have
    been created a moment ago by the entry handshake and not yet fetched
    anything, and reaping that would kill a session as it is being set up.
    """
    if last_active is None:
        return False
    if isinstance(last_active, bytes):
        last_active = last_active.decode(errors="replace")
    try:
        return (float(now) - float(last_active)) > float(stale_after)
    except (TypeError, ValueError):
        return False


def window_sustains_playback(window, target_duration,
                             min_segments=MIN_START_SEGMENTS):
    """True when a window is deep enough to hand to a player.

    Requires a segment count (players want a few segments listed before they
    will start) AND as much media as EXT-X-START tells the player to sit
    behind the live edge. Anything less and the playlist contradicts itself:
    the tag asks for a join point the window cannot reach, so the player
    starts at the window head with less runway than it was promised and is
    left absorbing its own fetch latency out of a buffer that was never
    deep enough. Measured with AVPlayer against a 6.4s window: it joined,
    lost lead steadily, and stalled eighteen seconds in.

    Only ever gates a COLD start. The window is trimmed to a fixed size and
    never shrinks once full, so a window too short to pass here is by
    definition one that has not filled yet. A mid-session reload is answered
    immediately - making a playing client wait is the very stall this exists
    to prevent.
    """
    if not window or len(window) < min_segments:
        return False
    try:
        needed = LIVE_EDGE_OFFSET_FACTOR * float(target_duration)
    except (TypeError, ValueError):
        needed = LIVE_EDGE_OFFSET_FACTOR * 4.0
    total = 0.0
    for entry in window:
        try:
            total += float(entry.get("dur") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
    return total >= needed


def render_media_playlist(window, target_duration, segment_name="{seq}.ts", adv_target=None):
    """
    Render an HLS media playlist (RFC 8216, version 3) from a window of
    segment descriptors: [{"seq": int, "dur": float, "disc": bool}, ...].
    Segment URIs are relative so they resolve against the playlist URL.

    ``adv_target`` is the manager's frozen EXT-X-TARGETDURATION; when supplied it
    is emitted verbatim so the value never changes across reloads (RFC 8216
    6.2.1). Without it (legacy descriptor) the per-window ceil is used.
    """
    # Frozen live-edge offset: ~2.5 config target-durations (~10s at the 4s
    # default) so the value is a session constant and never drifts across
    # reloads as the window slides (unlike a window-max derivation).
    start_offset = LIVE_EDGE_OFFSET_FACTOR * target_duration
    if not window:
        return (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXT-X-INDEPENDENT-SEGMENTS\n"
            # Ceil to match the populated branch; a fractional target must never
            # round DOWN below a real EXTINF (RFC 8216 4.3.3.1).
            f"#EXT-X-TARGETDURATION:{adv_target if adv_target else int(max(target_duration, 1) + 0.999)}\n"
            "#EXT-X-MEDIA-SEQUENCE:0\n"
        )
    # TARGETDURATION: prefer the manager's frozen constant. RFC 8216 6.2.1 forbids
    # it changing across reloads; a per-render ceil(window max) flaps on GOP
    # jitter, and AVPlayer latches the first value and stops advancing on a
    # contradiction. Legacy fallback keeps the ceil.
    advertised_target = adv_target if adv_target else int(max(entry["dur"] for entry in window) + 0.999)
    # Every segment begins on a keyframe and is prefixed with PAT/PMT
    # (_begin_segment), so each one decodes without reference to any other.
    # Declaring it lets a client seek or switch to any segment directly rather
    # than assuming it must decode from the window head. The one exception is
    # the mid-GOP force cut under max_segment_duration, which only fires in a
    # keyframe drought a healthy source never reaches; the tag stays worth
    # declaring for the seek behaviour it buys on every normal segment.
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        f"#EXT-X-TARGETDURATION:{advertised_target}",
        f"#EXT-X-MEDIA-SEQUENCE:{window[0]['seq']}",
    ]
    # EXT-X-START is emitted from the FIRST playlist onward. Its value is a
    # session constant (2.5 config target-durations), so it is stable across
    # reloads either way (RFC 8216 6.2.1) - but withholding it until the
    # window is deep enough removed the join hint from precisely the
    # cold-start reloads that need it, and made the tag set itself change
    # mid-session. A window shorter than the offset simply clamps the join
    # to the start of the window, which is what a player without the tag
    # already does, so nothing regresses on a thin window. It pins the join
    # point deterministically across players; a client that sets its own
    # offset still overrides it.
    lines.append(f"#EXT-X-START:TIME-OFFSET=-{start_offset:.3f},PRECISE=YES")
    for entry in window:
        if entry.get("disc"):
            lines.append("#EXT-X-DISCONTINUITY")
        lines.append(f"#EXTINF:{entry['dur']:.3f},")
        lines.append(segment_name.format(seq=entry["seq"]))
    return "\n".join(lines) + "\n"
