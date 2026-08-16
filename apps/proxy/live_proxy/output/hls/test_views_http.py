"""
HTTP-layer tests for the native HLS output views.

Exercises the REAL hls_playlist / hls_segment / hls_part views and the REAL
apps/proxy/urls.py routes through Django's test client. Only the heavy project
dependencies these three views never touch (ORM models, the proxy server,
websockets, stream generators) are stubbed into sys.modules before import, in
the same spirit as test_manager.py.

What this covers that the pure unit tests cannot:
  - URL routing, including p<seq>.<part>.ts vs <seq>.ts disambiguation
  - the _hls_cors decorator stacking correctly with DRF's api_view
  - OPTIONS preflight, and CORS headers on error responses
  - status codes and cache headers on every branch
  - the LL Blocking Playlist Reload directives (_HLS_msn/_HLS_part), including
    that a hold really holds and really observes media published mid-wait
  - end-to-end playlist rendering from a Redis descriptor

Needs Django + djangorestframework + gevent importable; skips cleanly if not.
Run standalone (same reasoning as test_manager: the sys.modules stubs would
poison any Django test sharing the process, and dispatcharr.test_discovery only
collects directories named "tests", so this file stays out of the Django
runner):

    python3 -m unittest apps.proxy.live_proxy.output.hls.test_views_http
"""

import json
import sys
import time
import types
import unittest
from pathlib import Path

# Skipped rather than raised at import: a module-level SkipTest aborts the
# whole run when this module is named explicitly alongside the dependency-free
# suites, instead of skipping just this one.
try:
    import django
    from django.conf import settings
    import rest_framework  # noqa: F401
    import gevent
    HAVE_DJANGO = True
    SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - environment-dependent
    HAVE_DJANGO = False
    SKIP_REASON = f"HTTP-layer tests need Django/DRF/gevent: {exc}"

requires_django = unittest.skipUnless(HAVE_DJANGO, SKIP_REASON)

# apps/proxy/live_proxy/output/hls/this_file.py -> repo root is 5 levels up.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------
# Fake Redis + stubs, installed before importing views
# --------------------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def exists(self, key):
        return 1 if key in self.store else 0

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)
        return len(keys)

    def hgetall(self, key):
        return self.store.get(key) or {}

    def hset(self, key, field=None, value=None, mapping=None):
        cur = self.store.setdefault(key, {})
        if mapping:
            cur.update(mapping)
        if field is not None:
            cur[field] = value
        return 1

    def expire(self, key, ttl):
        return key in self.store

    def sadd(self, key, *members):
        self.store.setdefault(key, set()).update(members)
        return len(members)

    def srem(self, key, *members):
        s = self.store.get(key)
        if isinstance(s, set):
            for m in members:
                s.discard(m)
        return len(members)

    def pipeline(self, transaction=False):
        return FakePipeline(self)


class FakePipeline:
    """Queues commands and applies them to the parent on execute(), which is
    all the views' _hls_touch_client needs."""

    def __init__(self, redis):
        self._redis = redis
        self._queued = []

    def __getattr__(self, name):
        def queue(*a, **k):
            self._queued.append((name, a, k))
            return self
        return queue

    def execute(self):
        out = []
        for name, a, k in self._queued:
            out.append(getattr(self._redis, name)(*a, **k))
        self._queued = []
        return out


BUFFER_REDIS = FakeRedis()
CLIENT_REDIS = FakeRedis()


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Modules this suite needs to be the REAL ones, but which test_manager replaces
# with fakes when both run in a single process. Dropping them from sys.modules
# forces a fresh import from disk; without this, whichever suite imported first
# wins and the other silently tests the wrong module (test_manager's fake
# RedisKeys has neither client_metadata nor output_buffer_chunk).
_MUST_BE_REAL = (
    "apps.proxy.live_proxy.redis_keys",
    "apps.proxy.live_proxy.config_helper",
    "apps.proxy.config",
)


def _restore_real_modules():
    """Drop fakes another suite installed, so the next import reads from disk.
    Only meaningful before this module imports the views."""
    for name in _MUST_BE_REAL:
        sys.modules.pop(name, None)


def _install_stubs():
    """(Re)register this suite's stubs. Called at import AND from setUp: the
    views import core.utils lazily *inside* hls_segment/hls_part, so a suite
    that loads later and re-stubs core.utils would otherwise hand those views a
    different fake Redis than the one the tests seed."""

    class _Dummy:
        """Permissive stand-in for an ORM model / arbitrary symbol."""
        objects = None

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, item):
            return _Dummy()

    def _noop(*a, **k):
        return None

    class ProxyServer:
        _instance = None

        def __init__(self):
            self.redis_client = CLIENT_REDIS

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    class RedisClient:
        @staticmethod
        def get_buffer():
            return BUFFER_REDIS

        @staticmethod
        def get_client():
            return CLIENT_REDIS

    class _Logger:
        def info(self, *a, **k):
            pass
        debug = warning = error = info

    # --- project modules the HLS views never exercise -----------------------
    _mod("apps.proxy.live_proxy.server", ProxyServer=ProxyServer)
    _mod("apps.proxy.live_proxy.channel_status",
         ChannelStatus=_Dummy, build_live_channel_stats_data=_noop)
    ts_pkg = _mod("apps.proxy.live_proxy.output.ts")
    ts_pkg.__path__ = []
    _mod("apps.proxy.live_proxy.output.ts.generator", create_stream_generator=_noop)
    fmp4_pkg = _mod("apps.proxy.live_proxy.output.fmp4")
    fmp4_pkg.__path__ = []
    _mod("apps.proxy.live_proxy.output.fmp4.generator", create_fmp4_stream_generator=_noop)
    _mod("apps.channels.models", Channel=_Dummy, Stream=_Dummy)
    _mod("apps.m3u.models", M3UAccount=_Dummy, M3UAccountProfile=_Dummy)
    _mod("apps.accounts.models", User=_Dummy)
    _mod("apps.accounts.permissions", IsAdmin=_Dummy,
         permission_classes_by_method=_noop, permission_classes_by_action=_noop)
    _mod("core.models", UserAgent=_Dummy, CoreSettings=_Dummy,
         PROXY_PROFILE_NAME="proxy")
    _mod("core.utils", RedisClient=RedisClient, send_websocket_update=_noop)
    _mod("apps.proxy.live_proxy.services.channel_service", ChannelService=_Dummy)
    _mod("apps.proxy.live_proxy.url_utils",
         generate_stream_url=_noop, transform_url=_noop,
         get_stream_info_for_switch=_noop, get_stream_object=_noop,
         get_alternate_streams=_noop)
    _mod("apps.proxy.utils", check_user_stream_limits=_noop)
    _mod("apps.proxy.stats_views", combined_stats=_noop)
    # get_client_ip moved here in upstream's client-IP/proxy-trust work
    # (c8b357c3); views.py imports both names from this module.
    _mod("dispatcharr.utils",
         network_access_allowed=lambda *a, **k: True,
         get_client_ip=lambda request: "127.0.0.1")

    # live_proxy.utils: real get_logger is Django-logging based; stub it, but
    # the module also carries get_client_ip which the views import.
    _mod("apps.proxy.live_proxy.utils",
         get_logger=lambda *a, **k: _Logger(),
         get_client_ip=lambda request: "127.0.0.1")

    # Sub-URLconfs pulled in by apps/proxy/urls.py via include().
    _mod("apps.proxy.live_proxy.urls", urlpatterns=[])
    _mod("apps.timeshift.urls", urlpatterns=[])
    _mod("apps.proxy.vod_proxy.urls", urlpatterns=[])


if HAVE_DJANGO:
    _restore_real_modules()
    _install_stubs()

    # --- Minimal Django config, then import the real views/urls ----------------
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="hls-http-test",
            ALLOWED_HOSTS=["*"],
            ROOT_URLCONF="_hls_test_urlconf",
            DATABASES={},
            INSTALLED_APPS=[
                "django.contrib.contenttypes",
                "django.contrib.auth",
                "rest_framework",
            ],
            REST_FRAMEWORK={"UNAUTHENTICATED_USER": None},
            USE_TZ=True,
        )

    django.setup()

    from django.urls import include, path  # noqa: E402

    # Mount the REAL apps/proxy/urls.py under /proxy/, as the project does.
    urlconf = _mod("_hls_test_urlconf")
    urlconf.urlpatterns = [path("proxy/", include("apps.proxy.urls"))]

    from django.test import Client  # noqa: E402
    from apps.proxy.live_proxy.redis_keys import RedisKeys  # noqa: E402

CH = "chan-http-test"
CID = "client-1"
TS_BYTES = b"\x47" * 188


def seed_client_record():
    """Minimum client record the views' _hls_touch_client needs."""
    CLIENT_REDIS.store[RedisKeys.client_metadata(CH, CID)] = {"output_format": "hls"}
    CLIENT_REDIS.store[RedisKeys.clients(CH)] = {CID}


def publish_descriptor(**over):
    desc = {
        "window": [
            {"seq": 5, "dur": 4.0, "disc": False, "pdt": "2026-07-01T00:00:00.000+00:00"},
            {"seq": 6, "dur": 4.0, "disc": False, "pdt": "2026-07-01T00:00:04.000+00:00"},
            {"seq": 7, "dur": 4.0, "disc": False, "pdt": "2026-07-01T00:00:08.000+00:00"},
        ],
        "target": 4,
        "adv_target": 4,
        "vcodec": "h264",
    }
    desc.update(over)
    CLIENT_REDIS.store[RedisKeys.output_playlist(CH, "hls")] = json.dumps(desc)
    return desc


def ll_descriptor(**over):
    base = {
        "part_target": 0.56,
        "parts": {"7": [[0.5, True], [0.5, False]]},
        "building": {"seq": 8, "parts": [[0.5, True]], "disc": False},
    }
    base.update(over)
    return publish_descriptor(**base)


class HLSHttpTestBase(unittest.TestCase):
    def setUp(self):
        # Re-assert our stubs: another stub-based suite in the same process may
        # have replaced core.utils since import (see _install_stubs).
        _install_stubs()
        BUFFER_REDIS.store.clear()
        CLIENT_REDIS.store.clear()
        seed_client_record()
        self.c = Client()

    def playlist_url(self, qs=""):
        return f"/proxy/hls/{CH}/{CID}/index.m3u8{qs}"

    def segment_url(self, seq):
        return f"/proxy/hls/{CH}/{CID}/{seq}.ts"

    def part_url(self, seq, part):
        return f"/proxy/hls/{CH}/{CID}/p{seq}.{part}.ts"


@requires_django
class RoutingTests(HLSHttpTestBase):
    def test_part_and_segment_routes_are_distinct(self):
        from django.urls import resolve
        self.assertEqual(resolve(self.segment_url(7)).url_name, "hls_segment")
        self.assertEqual(resolve(self.part_url(7, 2)).url_name, "hls_part")
        self.assertEqual(resolve(self.playlist_url()).url_name, "hls_playlist")

    def test_part_route_captures_seq_and_index(self):
        from django.urls import resolve
        m = resolve(self.part_url(12, 3))
        self.assertEqual(m.kwargs["seq"], 12)
        self.assertEqual(m.kwargs["part"], 3)


@requires_django
class CorsTests(HLSHttpTestBase):
    def test_options_preflight_on_playlist(self):
        r = self.c.options(self.playlist_url(), HTTP_ORIGIN="https://example.test")
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r["Access-Control-Allow-Origin"], "https://example.test")
        self.assertEqual(r["Vary"], "Origin")
        self.assertIn("GET", r["Access-Control-Allow-Methods"])

    def test_options_preflight_on_part(self):
        r = self.c.options(self.part_url(8, 0), HTTP_ORIGIN="https://example.test")
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r["Access-Control-Allow-Origin"], "https://example.test")

    def test_cors_header_on_success(self):
        publish_descriptor()
        r = self.c.get(self.playlist_url())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Access-Control-Allow-Origin"], "*")

    def test_cors_header_present_on_error_response(self):
        # A missing segment is the classic silent browser failure if the error
        # response omits CORS.
        r = self.c.get(self.segment_url(999))
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r["Access-Control-Allow-Origin"], "*")


@requires_django
class PlaylistTests(HLSHttpTestBase):
    def test_non_ll_playlist_rendered(self):
        publish_descriptor()
        r = self.c.get(self.playlist_url())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "application/vnd.apple.mpegurl")
        self.assertEqual(r["Cache-Control"], "no-cache")
        body = r.content.decode()
        self.assertIn("#EXT-X-VERSION:3", body)
        self.assertIn("#EXT-X-INDEPENDENT-SEGMENTS", body)
        self.assertIn("#EXT-X-TARGETDURATION:4", body)
        self.assertIn("#EXT-X-START:TIME-OFFSET=-", body)
        self.assertIn("#EXT-X-PROGRAM-DATE-TIME:", body)
        self.assertNotIn("#EXT-X-PART", body)

    def test_ll_playlist_rendered(self):
        ll_descriptor()
        body = self.c.get(self.playlist_url()).content.decode()
        self.assertIn("#EXT-X-VERSION:10", body)
        self.assertIn("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES", body)
        self.assertIn("#EXT-X-PART-INF:PART-TARGET=0.560", body)
        self.assertIn('#EXT-X-PART:DURATION=0.50000,URI="p7.0.ts",INDEPENDENT=YES', body)
        self.assertIn('#EXT-X-PRELOAD-HINT:TYPE=PART,URI="p8.1.ts"', body)
        self.assertNotIn("#EXT-X-START", body)

    def test_hevc_refused_with_415(self):
        publish_descriptor(vcodec="h265")
        r = self.c.get(self.playlist_url())
        self.assertEqual(r.status_code, 415)
        self.assertIn("H.264", r.json()["error"])

    def test_stopped_channel_returns_410(self):
        publish_descriptor()
        CLIENT_REDIS.store[RedisKeys.channel_stopping(CH)] = "1"
        self.assertEqual(self.c.get(self.playlist_url()).status_code, 410)


@requires_django
class BlockingReloadTests(HLSHttpTestBase):
    def test_part_without_msn_is_400(self):
        ll_descriptor()
        r = self.c.get(self.playlist_url("?_HLS_part=1"))
        self.assertEqual(r.status_code, 400)

    def test_non_integer_msn_is_400(self):
        ll_descriptor()
        self.assertEqual(self.c.get(self.playlist_url("?_HLS_msn=abc")).status_code, 400)

    def test_far_future_msn_is_400(self):
        # Client is out of sync; spec says trigger a full reload rather than
        # holding the connection forever.
        ll_descriptor()
        r = self.c.get(self.playlist_url("?_HLS_msn=99"))
        self.assertEqual(r.status_code, 400)

    def test_already_available_part_returns_immediately(self):
        ll_descriptor()
        started = time.time()
        r = self.c.get(self.playlist_url("?_HLS_msn=8&_HLS_part=0"))
        self.assertEqual(r.status_code, 200)
        self.assertLess(time.time() - started, 1.0)  # no hold
        self.assertIn("#EXT-X-VERSION:10", r.content.decode())

    def test_rolled_off_msn_returns_current_playlist(self):
        ll_descriptor()
        r = self.c.get(self.playlist_url("?_HLS_msn=1&_HLS_part=0"))
        self.assertEqual(r.status_code, 200)

    def test_unavailable_part_holds_then_503(self):
        # Deadline is min(3*target, 15) = 12s with target 4; shrink the target
        # in the descriptor so the hold is short but still real.
        ll_descriptor(target=0.05)
        started = time.time()
        r = self.c.get(self.playlist_url("?_HLS_msn=8&_HLS_part=5"))
        elapsed = time.time() - started
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r["Retry-After"], "1")
        self.assertGreater(elapsed, 0.05)   # it actually held
        self.assertLess(elapsed, 5.0)       # and it was bounded

    def test_msn_only_waits_for_segment_completion(self):
        # _HLS_msn without _HLS_part is satisfied only once that segment is
        # COMPLETE, not merely once its first part exists. Segment 8 is still
        # building, so this must not return 200 immediately.
        ll_descriptor(target=0.05)
        r = self.c.get(self.playlist_url("?_HLS_msn=8"))
        self.assertEqual(r.status_code, 503)

    def test_completed_msn_only_returns_200(self):
        ll_descriptor()
        r = self.c.get(self.playlist_url("?_HLS_msn=7"))
        self.assertEqual(r.status_code, 200)


@requires_django
class SegmentTests(HLSHttpTestBase):
    def test_segment_served_immutable(self):
        BUFFER_REDIS.store[RedisKeys.output_buffer_chunk(CH, "hls", 7)] = TS_BYTES
        r = self.c.get(self.segment_url(7))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "video/mp2t")
        self.assertEqual(r["Cache-Control"], "public, max-age=60, immutable")
        self.assertEqual(r.content, TS_BYTES)

    def test_expired_segment_404(self):
        self.assertEqual(self.c.get(self.segment_url(4)).status_code, 404)


@requires_django
class PartEndpointTests(HLSHttpTestBase):
    def test_existing_part_served(self):
        ll_descriptor()
        BUFFER_REDIS.store[RedisKeys.output_part(CH, "hls", 8, 0)] = TS_BYTES
        r = self.c.get(self.part_url(8, 0))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Type"], "video/mp2t")
        self.assertEqual(r["Cache-Control"], "public, max-age=15, immutable")
        self.assertEqual(r.content, TS_BYTES)

    def test_closed_segment_part_fast_404s(self):
        # Segment 7 is complete; a missing part of it can never appear, so this
        # must NOT burn the 3s blocking wait.
        ll_descriptor()
        started = time.time()
        r = self.c.get(self.part_url(7, 9))
        self.assertEqual(r.status_code, 404)
        self.assertLess(time.time() - started, 1.0)

    def test_building_part_blocks_then_404s(self):
        # A preload-hinted part of the BUILDING segment is worth waiting for;
        # when it never arrives the view still terminates at its deadline.
        ll_descriptor()
        started = time.time()
        r = self.c.get(self.part_url(8, 1))
        elapsed = time.time() - started
        self.assertEqual(r.status_code, 404)
        self.assertGreater(elapsed, 2.5)   # it held for the building segment
        self.assertLess(elapsed, 6.0)

    def test_building_part_served_when_it_appears_mid_hold(self):
        # The blocking wait must actually observe a part published while it
        # sleeps - that is the whole point of the preload hint.
        ll_descriptor()

        def publish_later():
            gevent.sleep(0.4)
            BUFFER_REDIS.store[RedisKeys.output_part(CH, "hls", 8, 1)] = TS_BYTES

        g = gevent.spawn(publish_later)
        r = self.c.get(self.part_url(8, 1))
        g.join()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, TS_BYTES)


if __name__ == "__main__":
    unittest.main()
