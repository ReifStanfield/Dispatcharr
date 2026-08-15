# Native HLS output for live channels

Dispatcharr can serve any live channel as a real HLS media playlist so that
HLS-native clients (Apple AVPlayer on iOS/tvOS/macOS, Safari, hls.js in the
browser, VLC, ffmpeg, QuickTime) can play it directly, with no client-side
remuxing. This document is the integration contract for client developers.

## Requesting HLS

A client asks for HLS on the normal stream endpoint using a standard query
parameter (no bespoke headers, no separate auth path):

```
GET /proxy/ts/stream/<channel_uuid>?output_format=hls
```

Aliases:

- `?output=hls` (XC-style parameter name) is accepted as well.
- Xtream Codes clients may use the `.m3u8` extension on the XC stream URL,
  e.g. `/live/<user>/<pass>/<id>.m3u8`, which resolves to the same output.

You can also set HLS as the server's **Default Output Format** (System >
Settings > Stream Settings), after which plain stream requests return HLS.
Clients that want a specific format should always set `output_format`
explicitly rather than relying on the server default.

## Redirect and playlist URLs

The stream request runs the normal init/auth/client-registration path and
then returns an **HTTP 302** to a client-scoped media playlist:

```
302 Location: /proxy/hls/<channel_uuid>/<client_id>/index.m3u8
```

Clients **must follow redirects** and **must preserve the redirected base
URL**. Segment URIs in the playlist are relative (`<seq>.ts`) and resolve
against the playlist URL, i.e. `/proxy/hls/<channel_uuid>/<client_id>/<seq>.ts`.
Do not hand-construct segment URLs against the pre-redirect `/proxy/ts/...`
path; fetch the redirected playlist and let the player resolve segments.

Every playlist and segment request touches the client record, so a polling
player keeps its session alive and a stopped player is reaped by the existing
ghost-client heartbeat. A client that stops fetching for roughly three
playlist reload intervals is treated as gone and its upstream slot released.

## Playlist shape

The media playlist is a standard live RFC 8216 (version 3) playlist:

- `#EXT-X-VERSION:3`
- `#EXT-X-INDEPENDENT-SEGMENTS`
- `#EXT-X-TARGETDURATION:<n>` (integer, a true upper bound on every segment)
- `#EXT-X-MEDIA-SEQUENCE:<n>` (monotonically increasing as segments roll off)
- `#EXT-X-START:TIME-OFFSET=-<n>,PRECISE=YES` pinning the join point behind
  the live edge, so every player starts with the same runway
- `#EXT-X-DISCONTINUITY` before a segment that follows a stream discontinuity
- `#EXT-X-PROGRAM-DATE-TIME` on each segment, anchoring it to the wall clock
- No `#EXT-X-ENDLIST` (the stream is live; players keep reloading)
- A rolling window of segments (default 10 x ~4s)

Segments are MPEG-TS, each prefixed with the current PAT and PMT so it decodes
independently. No transcoding or remuxing is performed; the source packets are
split on keyframe boundaries.

`TARGETDURATION` is computed once at session start and then **frozen** for the
life of the stream (RFC 8216 6.2.1 forbids it changing across reloads). It is
matched to the segmenter's cut target rather than padded above it: a player
reloads the playlist about once per `TARGETDURATION`, so advertising more than
a segment's worth makes the player reload more slowly than segments are
produced and it loses buffer lead every cycle until it stalls.

### Cold start

A channel that has just been tuned has no segments yet. The segmenter cuts its
first few segments at every keyframe and then ramps back to the full target,
so a playable playlist exists within about a GOP instead of several segment
durations. The playlist endpoint holds the first response (up to 20s) until
the window holds enough media to sustain playback, rather than handing a
player a one-segment playlist it will drain and stall on. An established
channel satisfies this on the first read and never waits.

## Low-Latency HLS

Low-Latency HLS is **off by default** and opted into by setting
`HLS_PART_TARGET` greater than zero (0.5 seconds is the suggested value). When
enabled, the media playlist becomes a Low-Latency HLS playlist (rfc8216bis)
and the in-progress segment is published as **partial segments** as it fills,
so a client can ride the live edge within ~1.5s instead of the ~3 target
durations a whole-segment live playlist forces.

It is opt-in rather than automatic because enabling it is not a transparent
addition: the playlist advertises `EXT-X-VERSION:10`, and a client that does
not implement version 10 is required by the spec to refuse the playlist
outright — it will *not* fall back to the whole segments the same playlist
still carries. Turn it on once you know the players on your deployment handle
it, and turn it back off (`HLS_PART_TARGET=0`) if one regresses.

The LL playlist additionally carries:

- `#EXT-X-VERSION:10`
- `#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=<3 x PART-TARGET>`
- `#EXT-X-PART-INF:PART-TARGET=<n>`
- `#EXT-X-PART:DURATION=<n>,URI="p<seq>.<part>.ts"` lines for the three most
  recent completed segments and for the segment currently being produced. The
  first part of each segment carries `INDEPENDENT=YES` (it begins on the
  keyframe and carries PAT/PMT).
- `#EXT-X-PRELOAD-HINT:TYPE=PART,URI="..."` naming the next part
- `#EXT-X-PROGRAM-DATE-TIME` on every segment (required by Apple's Low-Latency
  Server Configuration Profile; it also drives AVPlayer's
  `recommendedTimeOffsetFromLive`)

`EXT-X-START` is deliberately **not** emitted in LL mode: `PART-HOLD-BACK` is
the spec's native live-edge positioning and takes precedence.

`PART-TARGET` and `PART-HOLD-BACK`, like `TARGETDURATION`, are frozen for the
life of the stream. The advertised `PART-TARGET` is the configured target times
1.12, which keeps every real part at >=24fps inside the spec's 85% band while
covering cadence jitter.

### Blocking Playlist Reload

The playlist endpoint honours the `_HLS_msn` and `_HLS_part` delivery
directives (rfc8216bis 6.2.5.2). A request naming media the server has not
published yet is held open (gevent-cooperative, so it does not tie up a
worker) until that media appears, then answered with the fresh playlist.

- `_HLS_part` without `_HLS_msn` is malformed and returns **400**.
- An `_HLS_msn` far beyond the live edge means the client is out of sync and
  returns **400**, telling it to reload from scratch.
- If the hold deadline passes without the requested media appearing, the
  response is **503** with `Retry-After`, never a 200 missing what the client
  blocked on.
- Concurrent holds are bounded per channel (128); requests over the ceiling
  shed immediately with **503** rather than queueing.

### Partial segment URLs

Parts are served at `p<seq>.<part>.ts` relative to the playlist, i.e.
`/proxy/hls/<channel_uuid>/<client_id>/p<seq>.<part>.ts`, with
`Cache-Control: public, max-age=15, immutable`. A part of the in-progress
segment that has been preload-hinted but not yet stored blocks briefly (up to
3s) rather than 404-ing; a part of an already-closed segment 404s immediately.

Parts are short-lived in Redis — once a segment closes, the whole segment
serves any catch-up fetch.

### Non-LL clients

Leaving `HLS_PART_TARGET` at its default of 0 disables LL entirely and emits
the version-3 playlist described above — the path every HLS client supports.
This is also the setting to reach back for if a player regresses after LL is
enabled.

## MIME types, caching, and CORS

- Playlist responses use `Content-Type: application/vnd.apple.mpegurl` and
  `Cache-Control: no-cache` (the playlist changes on every reload).
- Segment responses use `Content-Type: video/mp2t` and
  `Cache-Control: public, max-age=60, immutable`. A finished segment is
  immutable — media-sequence numbers are monotonic and never reused — so a
  given `<seq>.ts` always maps to the same bytes and may be cached by
  browsers, hls.js, and any CDN in front of Dispatcharr.
- All HLS responses (playlist, segments, redirect, errors) carry permissive
  CORS headers and answer `OPTIONS` preflight, so a browser hls.js / Safari
  MSE player can fetch them cross-origin. Native players ignore CORS.

## Codec support

HLS output carries the source codec untouched in MPEG-TS segments.

- **H.264 (AVC) video is supported** and is broadly playable across all HLS
  clients.
- **HEVC / H.265 video is not served over HLS.** AVFoundation (AVPlayer,
  Safari) refuses HEVC in MPEG-TS, so serving it would black-screen those
  clients with no error. For an HEVC channel the playlist endpoint returns
  **HTTP 415** with a message directing the client to the MPEG-TS or fMP4
  output format. A client should treat a 415 on the playlist as "HLS not
  available for this channel" and fall back accordingly. HEVC-in-HLS would
  require fMP4/CMAF segments and is future work.
- Audio (AAC, AC-3, E-AC-3 including Atmos-as-JOC) is passed through and plays
  on clients that support it; Dolby audio bitstreams to a receiver on tvOS via
  the native player.

## Error responses

- **410 Gone** — the channel or this client's session was stopped. The player
  should re-enter via the stream URL rather than retrying the playlist.
- **415 Unsupported Media Type** — non-H.264 video (see Codec support).
- **404 Not Found** on a segment — it rolled out of the live window because
  the player fell too far behind.
- **503 Service Unavailable** with `Retry-After` — the channel is still
  starting and has produced no playlist yet.

## Server settings

- `HLS_SEGMENT_DURATION` (default 4 seconds) - target segment length.
- `HLS_WINDOW_SIZE` (default 10) - number of segments retained in the rolling
  live playlist.
- `HLS_PART_TARGET` (default 0, LL-HLS disabled) - Low-Latency partial-segment
  target. Set to ~0.5 seconds to enable LL-HLS; see the Low-Latency HLS section
  for the client-compatibility caveat.
