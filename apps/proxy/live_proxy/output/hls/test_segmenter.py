"""
Unit tests for the HLS TS segmenter. Dependency-free (stdlib unittest, no
Django/Redis), so they run standalone:

    python3 -m unittest apps.proxy.live_proxy.output.hls.test_segmenter
"""

import unittest

from .segmenter import (
    Part,
    Segment,
    TSSegmenter,
    TS_PACKET_SIZE,
    extract_pts,
    packet_pid,
    parse_pat,
    parse_pmt,
    render_media_playlist,
    window_sustains_playback,
    client_is_stale,
    client_stale_after,
    CLIENT_STALE_FLOOR,
    LIVE_EDGE_OFFSET_FACTOR,
    starts_keyframe,
)

VIDEO_PID = 256
PMT_PID = 4096
H264 = 0x1B


def make_packet(pid, payload, pusi=False, random_access=False):
    """Build one 188-byte TS packet with the given payload bytes."""
    header = bytearray(4)
    header[0] = 0x47
    header[1] = ((0x40 if pusi else 0x00) | (pid >> 8)) & 0xFF
    header[2] = pid & 0xFF

    if random_access:
        # adaptation field present + payload
        body_len = TS_PACKET_SIZE - 4 - 2 - len(payload)
        assert body_len >= 0, "payload too large for packet with AF"
        header[3] = 0x30  # AF + payload
        af = bytearray([1 + body_len, 0x40])  # af_length, RAI flag
        af.extend(b"\xff" * body_len)
        packet = bytes(header) + bytes(af) + bytes(payload)
    else:
        header[3] = 0x10  # payload only
        packet = bytes(header) + bytes(payload)
        packet += b"\xff" * (TS_PACKET_SIZE - len(packet))
    assert len(packet) == TS_PACKET_SIZE
    return packet


def make_pat():
    # pointer + table header (8 bytes from table_id) + one program entry
    payload = bytearray([0x00])                      # pointer_field
    payload += bytes([0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00])
    payload += bytes([0x00, 0x01, 0xE0 | (PMT_PID >> 8), PMT_PID & 0xFF])
    payload += bytes(4)                              # CRC placeholder
    return make_packet(0, payload, pusi=True)


def make_pmt():
    payload = bytearray([0x00])                      # pointer_field
    # table_id, section_length covers from after length to CRC
    es_loop = bytes([H264, 0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF, 0xF0, 0x00])
    section_length = 9 + len(es_loop) + 4            # post-length header + loop + CRC
    payload += bytes([0x02, 0xB0 | (section_length >> 8), section_length & 0xFF])
    payload += bytes([0x00, 0x01, 0xC1, 0x00, 0x00]) # tsid, ver, sec, last
    payload += bytes([0xE0 | (VIDEO_PID >> 8), VIDEO_PID & 0xFF, 0xF0, 0x00])  # PCR PID, prog info len
    payload += es_loop
    payload += bytes(4)                              # CRC placeholder
    return make_packet(PMT_PID, payload, pusi=True)


def make_video_pes(pts_seconds, keyframe, use_rai=False):
    """A PUSI video packet opening a PES with the given PTS."""
    pts = int(pts_seconds * 90000)
    p = bytearray()
    p += bytes([0x00, 0x00, 0x01, 0xE0, 0x00, 0x00])  # PES start, stream_id, length
    p += bytes([0x80, 0x80, 0x05])                    # flags, PTS-only, header len 5
    p += bytes([
        0x21 | (((pts >> 30) & 0x07) << 1),
        (pts >> 22) & 0xFF,
        0x01 | (((pts >> 15) & 0x7F) << 1),
        (pts >> 7) & 0xFF,
        0x01 | ((pts & 0x7F) << 1),
    ])
    # NAL start code + type
    if keyframe and not use_rai:
        p += bytes([0x00, 0x00, 0x00, 0x01, 0x65])    # IDR slice
    else:
        p += bytes([0x00, 0x00, 0x00, 0x01, 0x41])    # non-IDR slice
    return make_packet(VIDEO_PID, p, pusi=True, random_access=keyframe and use_rai)


def make_filler():
    return make_packet(VIDEO_PID, b"\x00" * 20)


class ParserTests(unittest.TestCase):
    def test_pat_pmt_roundtrip(self):
        self.assertEqual(parse_pat(make_pat()), PMT_PID)
        video_pid, stream_type = parse_pmt(make_pmt())
        self.assertEqual(video_pid, VIDEO_PID)
        self.assertEqual(stream_type, H264)

    def test_pts_roundtrip(self):
        packet = make_video_pes(1234.5, keyframe=True)
        self.assertAlmostEqual(extract_pts(packet), 1234.5, places=3)

    def test_keyframe_detection_nal_and_rai(self):
        self.assertTrue(starts_keyframe(make_video_pes(0, keyframe=True), H264))
        self.assertFalse(starts_keyframe(make_video_pes(0, keyframe=False), H264))
        self.assertTrue(starts_keyframe(make_video_pes(0, keyframe=True, use_rai=True), H264))

    def test_pid_extraction(self):
        self.assertEqual(packet_pid(make_pat()), 0)
        self.assertEqual(packet_pid(make_pmt()), PMT_PID)


def feed_stream(segmenter, gop_seconds, gop_count, start_pts=10.0, fillers_per_gop=5):
    """Feed `gop_count` GOPs of `gop_seconds` each; returns finished segments."""
    out = []
    for i in range(gop_count):
        pts = start_pts + i * gop_seconds
        out += segmenter.feed(make_video_pes(pts, keyframe=True))
        for j in range(fillers_per_gop):
            out += segmenter.feed(make_filler())
            out += segmenter.feed(make_video_pes(pts + (j + 1) * 0.2, keyframe=False))
    return out


class SegmenterTests(unittest.TestCase):
    def make_started(self, target=4.0, startup_cuts=0, ramp=()):
        # startup_cuts=0 / no ramp keeps most tests on steady-state behavior;
        # the fast-start ladder has its own dedicated tests.
        seg = TSSegmenter(target_duration=target, startup_keyframe_cuts=startup_cuts,
                          startup_ramp_fractions=ramp)
        seg.feed(make_pat())
        seg.feed(make_pmt())
        return seg

    def test_fast_start_ladder_cuts_first_segments_per_gop(self):
        seg = self.make_started(target=4.0, startup_cuts=3)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=9)
        durs = [round(s.duration, 3) for s in finished]
        # A cold channel accumulates media at live cadence, so the first
        # segments cut at EVERY keyframe (one 2s GOP each) to get a
        # playable window up fast; the normal 4s target then resumes.
        self.assertEqual(durs[:3], [2.0, 2.0, 2.0])
        self.assertTrue(all(abs(d - 4.0) < 0.01 for d in durs[3:]), durs)

    def test_fast_start_ramps_back_to_target(self):
        # The starter segments are cut from the pre-roll backlog, so they are
        # produced far faster than 1x; the first full-target segment after them
        # is the first produced at live cadence. Stepping straight from a 1-GOP
        # starter to a full 4s target puts that jump exactly where the player's
        # starter buffer runs dry. The ramp keeps the grain fine across the
        # handover: 0.5x then 0.75x of the target before steady state.
        seg = self.make_started(target=4.0, startup_cuts=3, ramp=(0.5, 0.75))
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=11)
        durs = [round(s.duration, 3) for s in finished]
        self.assertEqual(durs, [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0])

    def test_fast_start_ramp_never_exceeds_target(self):
        # Every ramp EXTINF must stay at or under the steady-state target, so
        # the frozen TARGETDURATION derived from it is never contradicted.
        seg = self.make_started(target=4.0, startup_cuts=4, ramp=(0.5, 0.75))
        finished = feed_stream(seg, gop_seconds=1.0, gop_count=24)
        self.assertTrue(all(s.duration <= 4.0 + 1e-6 for s in finished),
                        [s.duration for s in finished])

    def test_starter_floor_absorbs_short_spurious_keyframes(self):
        # A false-positive keyframe early in a real 2s GOP must not spend a
        # starter slot on a fragment. Observed live: starters of 0.97/1.03/
        # 0.30s against a 2.0s GOP, leaving the player 4.3s of runway where
        # the ladder was meant to hand it ~8s.
        #
        # The floor only claims to suppress cuts SHORTER than itself - a
        # spurious keyframe past 1.0s still cuts (the 1.03s starter above
        # would survive). Guaranteeing runway is the serving gate's job, not
        # this floor's; see window_sustains_playback.
        seg = self.make_started(target=4.0, startup_cuts=4, ramp=())
        out = []
        pts = 10.0
        for i in range(8):
            out += seg.feed(make_video_pes(pts, keyframe=True))          # real GOP
            out += seg.feed(make_video_pes(pts + 0.3, keyframe=True))    # spurious
            pts += 2.0
        durs = [round(s.duration, 3) for s in out]
        self.assertTrue(all(d >= 1.0 for d in durs), durs)
        # Every cut lands on a real GOP boundary, so the starters carry a
        # full 2s each rather than alternating with 0.3s fragments.
        self.assertTrue(all(abs(d - 2.0) < 0.01 for d in durs[:4]), durs)

    def test_starter_floor_off_restores_cut_at_every_keyframe(self):
        seg = self.make_started(target=4.0, startup_cuts=4, ramp=())
        seg.startup_min_duration = 0.0
        out = []
        pts = 10.0
        for i in range(4):
            out += seg.feed(make_video_pes(pts, keyframe=True))
            out += seg.feed(make_video_pes(pts + 0.3, keyframe=True))
            pts += 2.0
        self.assertTrue(any(round(s.duration, 3) == 0.3 for s in out),
                        [s.duration for s in out])

    def test_fast_start_ramp_can_be_disabled(self):
        # An empty ramp restores the original cliff: starters, then target.
        seg = self.make_started(target=4.0, startup_cuts=3, ramp=())
        durs = [round(s.duration, 3)
                for s in feed_stream(seg, gop_seconds=2.0, gop_count=9)]
        self.assertEqual(durs[:3], [2.0, 2.0, 2.0])
        self.assertTrue(all(abs(d - 4.0) < 0.01 for d in durs[3:]), durs)

    def test_cuts_on_keyframes_at_target_duration(self):
        seg = self.make_started(target=4.0)
        # 2-second GOPs: cuts must land every 2 GOPs (4.0s)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=7)
        self.assertEqual(len(finished), 3)
        for s in finished:
            self.assertAlmostEqual(s.duration, 4.0, places=3)

    def test_gop_dividing_the_target_still_cuts_at_the_target(self):
        # The common case must not regress: a 2s GOP against a 4s target keeps
        # cutting 4s segments. The threshold is target-minus-a-GOP plus an
        # epsilon (2.01 here), so the 2s keyframe is skipped and the 4s one
        # cuts - without the epsilon this would halve every segment.
        seg = self.make_started(target=4.0)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=13)
        durs = [round(s.duration, 3) for s in finished]
        self.assertTrue(all(abs(d - 4.0) < 0.01 for d in durs), durs)

    def test_gop_not_dividing_the_target_lands_under_it(self):
        # A 3s GOP used to cut at the first keyframe PAST the target - a 6s
        # segment, which is what forced TARGETDURATION up to 6 and made the
        # client reload slower than segments were produced. Aiming at the
        # largest whole number of GOPs that fits gives 3s segments instead:
        # keyframe-aligned, and under the target so it can be advertised as 4.
        seg = self.make_started(target=4.0)
        finished = feed_stream(seg, gop_seconds=3.0, gop_count=13)
        durs = [round(s.duration, 3) for s in finished[1:]]   # skip GOP learn-in
        self.assertTrue(durs, "expected segments")
        self.assertTrue(all(abs(d - 3.0) < 0.01 for d in durs), durs)
        self.assertTrue(all(d <= 4.0 for d in durs), durs)

    def test_every_extinf_rounds_to_the_advertised_target(self):
        # The invariant the whole TARGETDURATION change rests on (RFC 8216
        # 4.3.3.1: each EXTINF rounded to nearest int must be <= the target).
        #
        # Fed at a realistic 25fps, because the ceiling can only be enforced on
        # a picture boundary: the cut lands on the first frame at or past it,
        # so the emitted EXTINF overshoots by up to one frame interval. A live
        # source emitted 4.538s against a target of 4 - rounding to 5 - when
        # the ceiling sat at 4.49 and left a window narrower than one frame.
        for gop in (0.5, 1.0, 2.0, 3.0, 4.0, 4.6, 6.0):
            seg = self.make_started(target=4.0)
            seg.max_segment_duration = 4.0 + 0.35
            finished = []
            pts = 10.0
            # Enough GOPs to span several segments whatever the GOP length.
            for _ in range(max(8, int(30 / gop))):
                finished += seg.feed(make_video_pes(pts, keyframe=True))
                step = 0.04
                while step < gop:
                    finished += seg.feed(make_video_pes(pts + step, keyframe=False))
                    step += 0.04
                pts += gop
            self.assertTrue(finished, f"gop={gop} produced nothing")
            for s in finished:
                self.assertLessEqual(round(s.duration), 4,
                                     f"gop={gop} dur={s.duration}")

    def test_gop_longer_than_target_is_force_cut_at_the_ceiling(self):
        # A source whose GOP exceeds the target cannot give both keyframe
        # alignment and a segment under the target. The ceiling wins, so the
        # advertised value stays truthful; the cost is a mid-GOP cut.
        #
        # Fed by hand rather than through feed_stream: the force cut can only
        # fire on a picture whose PTS lands in [ceiling, next keyframe), and
        # feed_stream's fillers only run 1s into each GOP.
        seg = self.make_started(target=4.0)
        seg.max_segment_duration = 4.49
        finished = []
        pts = 10.0
        for _ in range(5):
            finished += seg.feed(make_video_pes(pts, keyframe=True))
            step = 0.5
            while step < 6.0:
                finished += seg.feed(make_video_pes(pts + step, keyframe=False))
                step += 0.5
            pts += 6.0
        self.assertTrue(finished)
        for s in finished:
            self.assertLessEqual(round(s.duration), 4, s.duration)

    def test_segments_start_with_pat_pmt(self):
        seg = self.make_started()
        finished = feed_stream(seg, gop_seconds=4.0, gop_count=3)
        self.assertGreaterEqual(len(finished), 1)
        for s in finished:
            self.assertEqual(s.data[0], 0x47)
            self.assertEqual(packet_pid(s.data[:TS_PACKET_SIZE]), 0)  # PAT first
            second = s.data[TS_PACKET_SIZE:2 * TS_PACKET_SIZE]
            self.assertEqual(packet_pid(second), PMT_PID)             # PMT second

    def test_no_segment_before_first_keyframe(self):
        seg = self.make_started()
        out = []
        out += seg.feed(make_video_pes(5.0, keyframe=False))
        out += seg.feed(make_filler())
        self.assertEqual(out, [])
        self.assertFalse(seg._collecting)

    def test_discontinuity_flag_propagates(self):
        seg = self.make_started(target=2.0)
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=2)
        finished += seg.flag_discontinuity()
        # Timeline jumps far ahead, as after a provider failover
        finished += feed_stream(seg, gop_seconds=2.0, gop_count=3, start_pts=9000.0)
        flagged = [s for s in finished if s.discontinuity]
        self.assertEqual(len(flagged), 1)

    def test_discontinuity_hard_cuts_open_segment(self):
        seg = self.make_started(target=4.0)
        # Open a segment and collect ~1s of frames without reaching the cut.
        seg.feed(make_video_pes(0.0, keyframe=True))
        for i in range(1, 4):
            seg.feed(make_video_pes(i * 0.5, keyframe=False))
        pre_gap_len = len(seg._current)
        self.assertTrue(seg._collecting)

        events = seg.flag_discontinuity()
        # The open segment is finished immediately from pre-gap bytes only,
        # with its measured span, and is NOT the discontinuity-tagged one.
        self.assertEqual(len(events), 1)
        tail = events[0]
        self.assertEqual(len(tail.data), pre_gap_len)
        self.assertAlmostEqual(tail.duration, 1.5, places=3)
        self.assertFalse(tail.discontinuity)
        self.assertFalse(seg._collecting)

        # Post-gap data before a keyframe is dropped; collection resumes at
        # the next keyframe, and THAT segment carries the discontinuity tag.
        out = seg.feed(make_video_pes(9000.2, keyframe=False))
        self.assertEqual(out, [])
        self.assertFalse(seg._collecting)
        seg.feed(make_video_pes(9001.0, keyframe=True))
        self.assertTrue(seg._collecting)
        out = seg.feed(make_video_pes(9006.0, keyframe=True))  # closes it
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].discontinuity)

    def test_discontinuity_discards_empty_open_segment(self):
        seg = self.make_started(target=4.0)
        # Only the opening keyframe collected: measured span is zero.
        seg.feed(make_video_pes(0.0, keyframe=True))
        self.assertEqual(seg.flag_discontinuity(), [])
        self.assertFalse(seg._collecting)
        # The tag still lands on the next started segment.
        seg.feed(make_video_pes(100.0, keyframe=True))
        out = seg.feed(make_video_pes(105.0, keyframe=True))
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].discontinuity)

    def test_resync_after_garbage(self):
        seg = self.make_started(target=2.0)
        seg.feed(b"\xde\xad\xbe\xef" * 33)  # garbage, not packet-aligned
        finished = feed_stream(seg, gop_seconds=2.0, gop_count=4)
        self.assertGreaterEqual(len(finished), 2)

    def test_pts_wrap_tolerated(self):
        seg = self.make_started(target=2.0)
        wrap_edge = (1 << 33) / 90000.0
        out = seg.feed(make_video_pes(wrap_edge - 1.0, keyframe=True))
        out += seg.feed(make_video_pes(1.0, keyframe=True))  # wrapped
        durations = [s.duration for s in out]
        for d in durations:
            self.assertGreater(d, 0)
            self.assertLessEqual(d, 8.0)


class PlaylistTests(unittest.TestCase):
    def test_render_basic(self):
        window = [
            {"seq": 7, "dur": 4.0, "disc": False},
            {"seq": 8, "dur": 4.2, "disc": False},
            {"seq": 9, "dur": 3.9, "disc": True},
        ]
        text = render_media_playlist(window, 4)
        self.assertIn("#EXTM3U", text)
        self.assertIn("#EXT-X-VERSION:3", text)
        self.assertIn("#EXT-X-TARGETDURATION:5", text)       # ceil(4.2)
        self.assertIn("#EXT-X-MEDIA-SEQUENCE:7", text)
        self.assertIn("#EXTINF:4.200,", text)
        self.assertIn("8.ts", text)
        self.assertNotIn("#EXT-X-ENDLIST", text)             # live
        # Live-edge start frozen at 2.5x the config target (2.5*4=10).
        self.assertIn("#EXT-X-START:TIME-OFFSET=-10.000,PRECISE=YES", text)
        # Discontinuity tag must precede its segment
        lines = text.splitlines()
        self.assertEqual(lines[lines.index("#EXT-X-DISCONTINUITY") + 2], "9.ts")

    def test_render_empty_window(self):
        text = render_media_playlist([], 4)
        self.assertIn("#EXT-X-MEDIA-SEQUENCE:0", text)
        self.assertIn("#EXT-X-TARGETDURATION:4", text)       # ceil(4)
        self.assertNotIn("#EXT-X-START", text)               # no segments to offset from

    def test_start_offset_present_on_a_thin_cold_start_window(self):
        # The join hint is emitted from the first playlist onward. Withholding
        # it until the window was deep enough removed it from precisely the
        # cold-start reloads that need it, and changed the tag set mid-session.
        # A window shorter than the offset just clamps the join to the window
        # start, which is what a player without the tag already does.
        thin = [{"seq": 0, "dur": 1.9, "disc": False}, {"seq": 1, "dur": 2.0, "disc": False}]
        text = render_media_playlist(thin, 4, adv_target=6)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-10.000,PRECISE=YES", text)

    def test_start_offset_identical_from_cold_start_to_steady_state(self):
        # RFC 8216 6.2.1 stability: the tag is byte-identical as the window
        # grows from the starter ladder to a full steady-state window.
        cold = [{"seq": 0, "dur": 1.9, "disc": False}]
        ramp = [{"seq": 0, "dur": 1.9, "disc": False}, {"seq": 1, "dur": 2.0, "disc": False},
                {"seq": 2, "dur": 4.0, "disc": False}]
        warm = [{"seq": i, "dur": 4.0, "disc": False} for i in range(10)]
        emitted = set()
        for w in (cold, ramp, warm):
            text = render_media_playlist(w, 4, adv_target=6)
            emitted.update(ln for ln in text.splitlines() if ln.startswith("#EXT-X-START"))
        self.assertEqual(len(emitted), 1)

    def test_targetduration_constant_across_window_shift(self):
        # RFC 8216 6.2.1: TARGETDURATION MUST NOT change across reloads. With a
        # frozen adv_target the emitted value is identical no matter how the
        # window's max EXTINF flaps across integer ceilings.
        adv = 8
        w1 = [{"seq": 1, "dur": 4.05, "disc": False}, {"seq": 2, "dur": 4.60, "disc": False}]
        w2 = [{"seq": 2, "dur": 4.60, "disc": False}, {"seq": 3, "dur": 6.46, "disc": False}]
        w3 = [{"seq": 3, "dur": 6.46, "disc": False}, {"seq": 4, "dur": 5.01, "disc": False}]
        tds = set()
        starts = set()
        for w in (w1, w2, w3):
            text = render_media_playlist(w, 4, adv_target=adv)
            td = [ln for ln in text.splitlines() if ln.startswith("#EXT-X-TARGETDURATION")]
            self.assertEqual(td, ["#EXT-X-TARGETDURATION:8"])
            tds.update(td)
            starts.update(ln for ln in text.splitlines() if ln.startswith("#EXT-X-START"))
            # TD must be >= every rounded EXTINF (RFC 8216 4.3.3.1).
            for e in w:
                self.assertLessEqual(round(e["dur"]), adv)
        self.assertEqual(len(tds), 1)      # never changed
        self.assertEqual(len(starts), 1)   # EXT-X-START also byte-stable


class StartWindowTests(unittest.TestCase):
    """The gate that decides when a cold-start playlist may be served."""

    def test_single_starter_segment_is_not_enough(self):
        # This is the shape that stalled players: one ~1.5s starter segment,
        # served the instant it existed.
        self.assertFalse(window_sustains_playback([{"seq": 0, "dur": 1.5}], 4))

    def test_three_short_starters_are_not_enough(self):
        # Three segments but only 3s of media: the player begins effectively at
        # the live edge and starves on the next cut.
        w = [{"seq": i, "dur": 1.0} for i in range(3)]
        self.assertFalse(window_sustains_playback(w, 4))

    def test_starter_ladder_of_three_gops_is_not_enough(self):
        # Three 2s GOP starters is 6s - enough to start playing, not enough to
        # absorb the player's own fetch latency, which is what the 10s join
        # offset exists to cover.
        w = [{"seq": i, "dur": 2.0} for i in range(3)]
        self.assertFalse(window_sustains_playback(w, 4))

    def test_gate_requires_the_advertised_join_offset(self):
        # The window AVPlayer stalled on: 6.4s, against an EXT-X-START that
        # tells the player to join 10s behind the live edge. The playlist was
        # promising a join point the window could not reach, so the player
        # started at the head with less runway than it was told to expect and
        # ran dry eighteen seconds in.
        w = [{"seq": 1, "dur": 2.619}, {"seq": 2, "dur": 2.002},
             {"seq": 3, "dur": 1.802}]
        self.assertLess(sum(e["dur"] for e in w), LIVE_EDGE_OFFSET_FACTOR * 4)
        self.assertFalse(window_sustains_playback(w, 4))

    def test_gate_and_playlist_agree_on_the_join_point(self):
        # The gate's threshold and EXT-X-START's offset are the same constant
        # by construction: a window that just clears the gate is exactly deep
        # enough to honor the offset the playlist advertises.
        target = 4
        w = [{"seq": i + 1, "dur": 2.5} for i in range(4)]   # 10.0s
        self.assertTrue(window_sustains_playback(w, target))
        text = render_media_playlist(w, target, adv_target=4)
        offset = LIVE_EDGE_OFFSET_FACTOR * target
        self.assertIn(f"#EXT-X-START:TIME-OFFSET=-{offset:.3f},PRECISE=YES", text)

    def test_first_session_window_starts_at_seq_one(self):
        # The chunk index backing the media sequence starts at 1, not 0, so
        # nothing here may treat a nonzero first seq as proof the window has
        # rolled - that read served a cold 1-segment playlist unguarded.
        self.assertFalse(window_sustains_playback([{"seq": 1, "dur": 2.0}], 4))

    def test_two_long_segments_still_wait_for_a_third(self):
        # Duration alone is not sufficient; players want a few segments listed.
        self.assertGreaterEqual(5.0 + 6.0, LIVE_EDGE_OFFSET_FACTOR * 4)
        self.assertFalse(
            window_sustains_playback([{"seq": 0, "dur": 5.0}, {"seq": 1, "dur": 6.0}], 4)
        )

    def test_rolled_window_is_never_gated(self):
        # Mid-session the window is full and slides, so it always clears both
        # conditions and a reload is answered immediately - blocking one would
        # stall a playing client, the exact failure this gate exists to
        # prevent. A high first seq is NOT what establishes this (sequences
        # start at 1, so that test would pass on the very first segment); a
        # window too short to clear the gate is by definition one that has not
        # filled yet.
        rolled = [{"seq": 40 + i, "dur": 4.0} for i in range(10)]
        self.assertTrue(window_sustains_playback(rolled, 4))

    def test_warm_window_passes_immediately(self):
        w = [{"seq": i, "dur": 4.0} for i in range(10)]
        self.assertTrue(window_sustains_playback(w, 4))

    def test_empty_and_malformed_windows_do_not_raise(self):
        self.assertFalse(window_sustains_playback([], 4))
        self.assertFalse(window_sustains_playback(None, 4))
        # A descriptor missing or corrupting "dur" must not 500 the endpoint;
        # it simply contributes nothing to the measured depth.
        w = [{"seq": 0}, {"seq": 1, "dur": None}, {"seq": 2, "dur": "x"}]
        self.assertFalse(window_sustains_playback(w, 4))

    def test_non_numeric_target_falls_back(self):
        # Falls back to the 4s default target, so the requirement is the same
        # 10s it would be with an explicit 4.
        w = [{"seq": i, "dur": 2.0} for i in range(3)]     # 6.0s
        self.assertFalse(window_sustains_playback(w, None))
        w = [{"seq": i, "dur": 4.0} for i in range(3)]     # 12.0s
        self.assertTrue(window_sustains_playback(w, None))


class ClientStalenessTests(unittest.TestCase):
    """When a pull-based client is judged gone, releasing its upstream slot."""

    def test_stale_window_is_three_polls_with_a_floor(self):
        # A player reloads about once per TARGETDURATION, which now equals the
        # cut target, so three missed reloads is unambiguous.
        self.assertEqual(client_stale_after(10), 30.0)
        # ...but a short target must not let one slow moment reap a client.
        self.assertEqual(client_stale_after(4), CLIENT_STALE_FLOOR)
        self.assertEqual(client_stale_after(1), CLIENT_STALE_FLOOR)

    def test_stale_window_survives_a_junk_target(self):
        self.assertEqual(client_stale_after(None), CLIENT_STALE_FLOOR)
        self.assertEqual(client_stale_after("x"), CLIENT_STALE_FLOOR)

    def test_recent_fetch_is_not_stale(self):
        now = 1000.0
        self.assertFalse(client_is_stale(str(now - 4), now, 15))
        self.assertFalse(client_is_stale(str(now - 14.9), now, 15))

    def test_missed_polls_are_stale(self):
        now = 1000.0
        self.assertTrue(client_is_stale(str(now - 15.1), now, 15))
        self.assertTrue(client_is_stale(str(now - 300), now, 15))

    def test_redis_returns_bytes(self):
        now = 1000.0
        self.assertTrue(client_is_stale(b"985.0", now, 14))
        self.assertFalse(client_is_stale(b"999.0", now, 14))

    def test_unreadable_timestamp_is_never_stale(self):
        # A record created by the entry handshake that has not fetched
        # anything yet must not be reaped as the session is being set up.
        now = 1000.0
        self.assertFalse(client_is_stale(None, now, 15))
        self.assertFalse(client_is_stale("", now, 15))
        self.assertFalse(client_is_stale("not-a-number", now, 15))
        self.assertFalse(client_is_stale(b"\xff\xfe", now, 15))

    def test_clock_skew_does_not_reap(self):
        # A timestamp in the future (worker clock drift) reads as very recent,
        # never as stale.
        now = 1000.0
        self.assertFalse(client_is_stale(str(now + 60), now, 15))


class PartTests(unittest.TestCase):
    """Low-Latency HLS partial segments."""

    def make_ll(self, target=4.0, part_target=0.5, **kwargs):
        # startup_cuts=0 / no ramp keeps these on steady-state cut behavior;
        # the fast-start ladder has its own dedicated tests.
        seg = TSSegmenter(target_duration=target, part_target=part_target,
                          startup_keyframe_cuts=0, startup_ramp_fractions=(),
                          **kwargs)
        seg.feed(make_pat())
        seg.feed(make_pmt())
        return seg

    def _feed(self, seg, frames):
        events = []
        for pts, keyframe in frames:
            events.extend(seg.feed(make_video_pes(pts, keyframe=keyframe)))
        return events

    def test_parts_tile_the_segment(self):
        seg = self.make_ll()
        frames = [(0.0, True)]
        t = 0.25
        while t < 4.0:
            frames.append((round(t, 2), False))
            t += 0.25
        frames.append((4.0, True))  # keyframe at/after target cuts the segment
        events = self._feed(seg, frames)

        parts = [e for e in events if isinstance(e, Part)]
        segments = [e for e in events if isinstance(e, Segment)]
        self.assertEqual(len(segments), 1)
        # ~8 parts of ~0.5s span the 4s segment (including the final tail part).
        self.assertGreaterEqual(len(parts), 6)
        # Only the first part carries the keyframe + PAT/PMT.
        self.assertTrue(parts[0].independent)
        self.assertFalse(parts[1].independent)
        # A segment's parts concatenate exactly to the segment bytes.
        self.assertEqual(b"".join(p.data for p in parts), segments[0].data)
        for p in parts:
            self.assertLessEqual(p.duration, 2 * 0.5)  # bounded near part_target

    def test_part_target_zero_emits_only_segments(self):
        seg = self.make_ll(part_target=0.0)
        frames = [(0.0, True), (2.0, False), (4.0, True)]
        events = self._feed(seg, frames)
        self.assertTrue(all(isinstance(e, Segment) for e in events))
        self.assertEqual(len(events), 1)

    def test_bframe_reorder_gives_no_garbage_durations(self):
        # Decode-order PTS that dips below the previous frame like B-frame
        # reordering. A naive pts-start delta goes slightly negative and, if
        # treated as a 33-bit wrap, yields a ~95443s garbage part duration that
        # makes AVPlayer reject the playlist. Durations must stay sane.
        seg = self.make_ll()
        frames = [(0.0, True)]
        base = 0.0
        while base < 4.0:
            base += 0.25
            frames.append((round(base + 0.1, 3), False))  # ahead in presentation
            frames.append((round(base, 3), False))        # dips back (B-frame)
        frames.append((4.2, True))  # keyframe cuts the segment
        events = self._feed(seg, frames)
        parts = [e for e in events if isinstance(e, Part)]
        self.assertTrue(parts)
        for p in parts:
            self.assertGreater(p.duration, 0)
            self.assertLess(p.duration, seg.target_duration)  # never ~95443s

    def test_part_ceiling_clamps_over_target_tail(self):
        # A coarse frame cadence makes a tail longer than the advertised ceiling;
        # every emitted part duration must still be <= the ceiling so no
        # EXT-X-PART exceeds the frozen PART-TARGET (RFC 8216bis 4.4.4.9).
        ceiling = 0.56
        seg = self.make_ll(part_ceiling=ceiling)
        # ~1.5fps: PES boundaries 0.7s apart, so a raw tail would reach ~0.7s.
        frames = [(0.0, True), (0.7, False), (1.4, False), (2.1, False),
                  (2.8, False), (3.5, False), (4.2, True)]
        events = self._feed(seg, frames)
        parts = [e for e in events if isinstance(e, Part)]
        self.assertTrue(parts)
        for p in parts:
            self.assertLessEqual(p.duration, ceiling + 1e-9)

    def test_non_final_parts_land_in_85_percent_band(self):
        # At a realistic frame cadence the frozen adv_part (part_target * 1.12)
        # keeps every non-final part inside [0.85*adv_part, adv_part]
        # (RFC 8216bis 4.4.4.9), the last part of each segment exempt.
        part_target = 0.5
        adv_part = round(part_target * 1.12, 3)   # 0.56, as the manager freezes
        seg = self.make_ll(part_target=part_target, part_ceiling=adv_part)
        frames = [(0.0, True)]
        i = 1
        while i / 30.0 < 8.0:                     # 30fps, two 4s segments
            t = round(i / 30.0, 5)
            kf = abs(t - 4.0) < 1e-6              # a keyframe at the 4s target
            frames.append((t, kf))
            i += 1
        events = self._feed(seg, frames)
        # Split parts by the Segment they precede; the last part before each
        # Segment is that segment's final (exempt) part.
        groups, cur = [], []
        for e in events:
            if isinstance(e, Part):
                cur.append(e)
            elif isinstance(e, Segment):
                groups.append(cur)
                cur = []
        self.assertTrue(any(len(g) >= 2 for g in groups))
        for g in groups:
            for p in g[:-1]:                      # non-final parts
                self.assertLessEqual(p.duration, adv_part)
                self.assertGreaterEqual(p.duration, 0.85 * adv_part)
            if g:
                self.assertLessEqual(g[-1].duration, adv_part)  # final part

    def test_discontinuity_flushes_final_part_before_segment(self):
        # A hard cut at a discontinuity must still tile the closing segment:
        # its trailing bytes flush as a final Part, ordered before the Segment.
        seg = self.make_ll()
        seg.feed(make_video_pes(0.0, keyframe=True))
        for t in (0.25, 0.5, 0.75, 1.0, 1.25):
            seg.feed(make_video_pes(t, keyframe=False))
        events = seg.flag_discontinuity()
        self.assertTrue(events)
        self.assertIsInstance(events[-1], Segment)
        parts = [e for e in events if isinstance(e, Part)]
        self.assertTrue(parts)
        # Parts emitted during the segment plus this tail cover the whole segment.
        self.assertFalse(events[-1].discontinuity)  # the NEXT segment carries it

    def test_parts_suppressed_until_keyframe_reanchors_after_discontinuity(self):
        # Post-gap bytes before a keyframe must not be emitted as parts against
        # the pre-jump part start, which would advertise a garbage duration.
        seg = self.make_ll()
        seg.feed(make_video_pes(0.0, keyframe=True))
        seg.feed(make_video_pes(0.6, keyframe=False))
        seg.flag_discontinuity()
        events = seg.feed(make_video_pes(9000.6, keyframe=False))
        self.assertEqual(events, [])


class LowLatencyPlaylistTests(unittest.TestCase):
    def test_render_low_latency(self):
        window = [
            {"seq": 5, "dur": 4.0, "disc": False},
            {"seq": 6, "dur": 4.1, "disc": False},
        ]
        parts_by_seq = {"6": [[0.5, True], [0.5, False]]}
        building = {"seq": 7, "parts": [[0.5, True], [0.4, False]]}
        # part_target here is the FROZEN advertised PART-TARGET (adv_part), which
        # the manager computes and carries in the descriptor; render emits it and
        # PART-HOLD-BACK = 3x it verbatim, recomputing neither.
        text = render_media_playlist(
            window, 4, part_target=0.56, parts_by_seq=parts_by_seq,
            building=building, adv_target=8,
        )
        self.assertIn("#EXT-X-VERSION:10", text)
        self.assertIn("#EXT-X-INDEPENDENT-SEGMENTS", text)
        self.assertIn("#EXT-X-TARGETDURATION:8", text)  # frozen, not ceil(4.1)=5
        self.assertIn(
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=1.680", text)
        self.assertIn("#EXT-X-PART-INF:PART-TARGET=0.560", text)
        # LL never carries EXT-X-START: PART-HOLD-BACK positions the client.
        self.assertNotIn("#EXT-X-START", text)
        # Parts of the last completed segment, with the first marked independent.
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p6.0.ts",INDEPENDENT=YES', text)
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p6.1.ts"', text)
        # In-progress segment's parts + a preload hint for the next part.
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p7.0.ts",INDEPENDENT=YES', text)
        self.assertIn('#EXT-X-PRELOAD-HINT:TYPE=PART,URI="p7.2.ts"', text)
        # The completed segments are still present for non-LL clients.
        self.assertIn("6.ts", text)
        # Same call without part_target stays a plain version-3 playlist that
        # keeps this branch's EXT-X-START join hint.
        plain = render_media_playlist(window, 4)
        self.assertIn("#EXT-X-VERSION:3", plain)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-", plain)
        self.assertNotIn("#EXT-X-PART", plain)

    def test_part_target_and_holdback_constant(self):
        # PART-TARGET / PART-HOLD-BACK are frozen constants: render must emit the
        # passed adv_part verbatim regardless of the listed part maxima (0.501 vs
        # 0.534), so AVPlayer's blocking-reload timing model never sees them flap.
        w = [{"seq": 6, "dur": 4.1}]
        t1 = render_media_playlist(
            w, 4, part_target=0.56, parts_by_seq={"6": [[0.501, True]]},
            building={"seq": 7, "parts": [[0.501, True]]}, adv_target=8)
        t2 = render_media_playlist(
            w, 4, part_target=0.56, parts_by_seq={"6": [[0.534, True]]},
            building={"seq": 7, "parts": [[0.534, True]]}, adv_target=8)
        for text in (t1, t2):
            self.assertIn("#EXT-X-PART-INF:PART-TARGET=0.560", text)
            self.assertIn("PART-HOLD-BACK=1.680", text)

    def test_ll_program_date_time_and_building_discontinuity(self):
        # Apple's LL profile requires EXT-X-PROGRAM-DATE-TIME per segment; the
        # building segment's discontinuity must be signalled before its first
        # part (never inserted retroactively).
        window = [
            {"seq": 5, "dur": 4.0, "disc": False, "pdt": "2026-07-01T00:00:00.000+00:00"},
            {"seq": 6, "dur": 4.1, "disc": True, "pdt": "2026-07-01T00:00:04.000+00:00"},
        ]
        building = {"seq": 7, "parts": [[0.5, True]], "disc": True}
        text = render_media_playlist(
            window, 4, part_target=0.56,
            parts_by_seq={"6": [[0.5, True]]}, building=building, adv_target=8)
        lines = text.splitlines()
        # Segment 5 has no listed parts, so its PDT sits immediately before EXTINF.
        pdt5 = lines.index("#EXT-X-PROGRAM-DATE-TIME:2026-07-01T00:00:00.000+00:00")
        self.assertEqual(lines[pdt5 + 1], "#EXTINF:4.000,")
        # Segment 6's PDT precedes its EXTINF (parts intervene) and follows the
        # discontinuity tag for that segment.
        pdt6 = lines.index("#EXT-X-PROGRAM-DATE-TIME:2026-07-01T00:00:04.000+00:00")
        extinf6 = lines.index("#EXTINF:4.100,")
        self.assertLess(pdt6, extinf6)
        disc_positions = [i for i, l in enumerate(lines) if l == "#EXT-X-DISCONTINUITY"]
        self.assertTrue(any(i < pdt6 for i in disc_positions))
        # Building discontinuity precedes the building segment's first part line.
        first_building_part = lines.index(
            '#EXT-X-PART:DURATION=0.50000,URI="p7.0.ts",INDEPENDENT=YES')
        self.assertTrue(any(i < first_building_part for i in disc_positions))

    def test_only_recent_segments_carry_part_lines(self):
        # Parts are rendered for the last PARTS_RENDERED_SEGMENTS only; older
        # segments stay whole-segment, which is all a client that far behind
        # the live edge needs.
        window = [{"seq": s, "dur": 4.0} for s in range(1, 7)]
        parts_by_seq = {str(s): [[0.5, True]] for s in range(1, 7)}
        text = render_media_playlist(
            window, 4, part_target=0.56, parts_by_seq=parts_by_seq, adv_target=8)
        self.assertNotIn('URI="p1.0.ts"', text)
        self.assertNotIn('URI="p3.0.ts"', text)
        self.assertIn('URI="p4.0.ts"', text)
        self.assertIn('URI="p6.0.ts"', text)


class VideoCodecTests(unittest.TestCase):
    def test_video_codec_learned_from_pmt(self):
        seg = TSSegmenter(target_duration=4.0)
        self.assertIsNone(seg.video_codec)
        seg.feed(make_pat())
        seg.feed(make_pmt())
        self.assertEqual(seg.video_codec, "h264")


if __name__ == "__main__":
    unittest.main()
