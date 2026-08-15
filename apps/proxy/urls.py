from django.urls import path, include

from apps.proxy import stats_views
from apps.proxy.live_proxy.views import hls_playlist, hls_segment, hls_part

app_name = 'proxy'

urlpatterns = [
    path('stats/', stats_views.combined_stats, name='combined_stats'),
    path('ts/', include('apps.proxy.live_proxy.urls')),
    path('catchup/', include('apps.timeshift.urls')),
    # Native HLS output for live channels (served by live_proxy).
    path('hls/<str:channel_id>/<str:client_id>/index.m3u8', hls_playlist, name='hls_playlist'),
    path('hls/<str:channel_id>/<str:client_id>/<int:seq>.ts', hls_segment, name='hls_segment'),
    # Low-Latency HLS partial segment: p<seq>.<part>.ts. The literal "p" prefix
    # keeps it distinct from the <int:seq>.ts segment route above.
    path('hls/<str:channel_id>/<str:client_id>/p<int:seq>.<int:part>.ts', hls_part, name='hls_part'),
    path('vod/', include('apps.proxy.vod_proxy.urls')),
]
