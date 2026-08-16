"""
features.py — IP-window behavioral features for Bot/WebAttack separation

Bot vs WebAttack flow-stat uzayinda ayrismiyor (her ikisi de HTTP/TCP).
Ayirim zamansal davranis deseninde:
  Bot:         periyodik C2 beacon, sabit interval, dusuk entropy
  WebAttack:  rastgele timing, yuksek payload cesitliligi, yuksek entropy

Bu modul, src_ip bazinda 40-akislik pencereden 8 ozellik hesaplar.
"""

import numpy as np
from collections import Counter
import math


def _entropy(values):
    if not values:
        return 0.0
    n = len(values)
    counter = Counter(values)
    return -sum((c / n) * math.log2(c / n) for c in counter.values())


def compute_ip_window_features(metadata_list):
    """
    metadata_list: list of dicts, her biri:
        - ts: datetime
        - dest_ip: str
        - dest_port: int
        - total_bytes: int
        - dns_queries: list[str] (DNS rrname)
        - http_uri: str or None
        - tls_sni: str or None

    Returns: np.ndarray (8,) float32
    """
    n = len(metadata_list)
    if n < 2:
        return np.zeros(8, dtype=np.float32)

    # 1-2. Beacon interval: Bot periyodik C2 callback, WebAttack rastgele
    timestamps = [m["ts"] for m in metadata_list]
    intervals = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if delta > 0:
            intervals.append(delta)

    if intervals:
        beacon_mean = float(np.mean(intervals))
        beacon_std  = float(np.std(intervals))
    else:
        beacon_mean = 0.0
        beacon_std  = 0.0

    # 3. dst_ip_entropy: Bot ayni C2'ye gider (düsük), WebAttack tek hedef (düsük/orta)
    dst_ips = [m["dest_ip"] for m in metadata_list]
    dst_ip_entropy = _entropy(dst_ips)

    # 4. unique_dns_per_min: Bot DGA/resolve trafigi (yüksek), WebAttack DNS yok
    all_dns = []
    for m in metadata_list:
        all_dns.extend(m.get("dns_queries") or [])
    unique_dns = len(set(all_dns)) if all_dns else 0
    total_sec = max(sum(intervals), 1.0) if intervals else 1.0
    dns_per_min = unique_dns / (total_sec / 60.0)

    # 5. http_uri_entropy: Bot sabit callback path (düsük), WebAttack SQLi/XSS payload (yüksek)
    uris = [m["http_uri"] for m in metadata_list if m.get("http_uri")]
    uri_entropy = _entropy(uris) if uris else 0.0

    # 6. same_dst_port_ratio: Bot C2 portu sabit (yüksek)
    ports = [m["dest_port"] for m in metadata_list]
    if ports:
        top_port_count = Counter(ports).most_common(1)[0][1]
        same_port_ratio = top_port_count / len(ports)
    else:
        same_port_ratio = 0.0

    # 7. tls_sni_reuse_ratio: Bot ayni C2 domain (yüksek)
    snis = [m["tls_sni"] for m in metadata_list if m.get("tls_sni")]
    if snis:
        tls_sni_reuse = 1.0 - (len(set(snis)) / max(len(snis), 1))
    else:
        tls_sni_reuse = 0.0

    # 8. payload_size_variance: Bot heartbeat sabit (düsük), WebAttack injection (yüksek)
    sizes = [m["total_bytes"] for m in metadata_list]
    pkt_var = float(np.var(sizes)) if len(sizes) > 1 else 0.0

    return np.array([
        beacon_mean, beacon_std,
        dst_ip_entropy, dns_per_min,
        uri_entropy, same_port_ratio,
        tls_sni_reuse, pkt_var,
    ], dtype=np.float32)
