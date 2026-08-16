#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone feature extraction for CORAL evaluation - no tensorflow dependency
"""

import numpy as np
from dateutil.parser import isoparse


def _extract_tcp_flags(event):
    tcp = event.get("tcp")
    if not tcp:
        return {"syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0}
    return {
        "syn": float(tcp.get("syn", 0) or 0),
        "ack": float(tcp.get("ack", 0) or 0),
        "fin": float(tcp.get("fin", 0) or 0),
        "rst": float(tcp.get("rst", 0) or 0),
        "psh": float(tcp.get("psh", 0) or 0),
    }


def _extract_http_features(enrichment):
    if not enrichment or "http" not in enrichment:
        return {"mo": [0,0,0,0], "so": [0,0,0,0], "cth": 0, "cto": 0, "ctj": 0, "ct_": 0}
    http = enrichment["http"].get("http", {})
    method = (http.get("http_method") or "").upper()
    status = http.get("status", 0) or 0
    ctype  = (http.get("content_type") or "").lower()
    mo = [float(method=="GET"), float(method=="POST"), float(method=="HEAD"), 
          float(method not in ("GET","POST","HEAD"))]
    so = [float(200<=status<300), float(300<=status<400), float(400<=status<500), float(500<=status<600)]
    return {"mo": mo, "so": so, "cth": float("html" in ctype), "cto": float("octet" in ctype),
            "ctj": float("json" in ctype), "ct_": float(ctype not in ("html","octet","json",""))}


def _extract_tls_features(enrichment):
    if not enrichment or "tls" not in enrichment:
        return {"e": 0, "sni": 0}
    tls = enrichment["tls"].get("tls", {})
    return {"e": 1, "sni": float(bool(tls.get("sni", "")))}


def _extract_dns_features(enrichment):
    if not enrichment or "dns" not in enrichment:
        return {"e": 0, "qa": 0, "qaa": 0, "qm": 0, "qo": 0, "rn": 0, "rnx": 0, "rr": 0, "ro": 0}
    dns = enrichment["dns"].get("dns", {})
    qt = set((q.get("rrtype") or "").upper() for q in (dns.get("queries") or []))
    rc = (dns.get("rcode") or "").upper()
    return {
        "e": 1,
        "qa": float("A" in qt), "qaa": float("AAAA" in qt),
        "qm": float("MX" in qt), "qo": float(bool(qt - {"A","AAAA","MX"})),
        "rn": float(rc == "NOERROR"), "rnx": float(rc == "NXDOMAIN"),
        "rr": float(rc == "REFUSED"), "ro": float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED")),
    }


def extract_features_v7_standalone(event, enrichment=None):
    """
    Suricata flow event'inden 70 ozellik cikarir (standalone - no TF dependency).
    Donus: np.ndarray shape (70,) veya None
    """
    if event.get("event_type") != "flow":
        return None

    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)

    try:
        t0  = isoparse((flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1  = isoparse((flow.get("end") or "").replace("Z","+00:00"))
        dur = max((t1 - t0).total_seconds(), 0.0)
    except Exception:
        dur = 0.0

    dp  = int(event.get("dest_port", 0) or 0)
    sp  = int(event.get("src_port",  0) or 0)
    tp  = pts + ptc
    tb  = bts + btc
    sd  = max(dur, 0.1);  spk = max(tp, 1);  sb = max(tb, 1)

    proto     = (event.get("proto")     or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)
    state     = (flow.get("state")  or "").lower()
    reason    = (flow.get("reason") or "").lower()
    age       = float(flow.get("age", 0) or 0)

    # 42 temel ozellik
    base = np.array([
        dur, pts, ptc, bts, btc, tp, tb,
        tp/sd, tb/sd, tb/spk,
        bts/sb, pts/spk, abs(bts - btc)/sb,
        btc/sb, ptc/spk, age,
        float(ptc == 0),
        abs((bts / max(pts,1)) - (btc / max(ptc,1))),
        dp, sp,
        float(dp < 1024), float(1024 <= dp < 49152), float(dp >= 49152),
        float(sp == dp),
        float(ip_v == 6),
        float(proto == "TCP"), float(proto == "UDP"),
        float(proto in ("ICMP","ICMPv6")),
        float(app_proto == "http"), float(app_proto == "dns"),
        float(app_proto == "tls"),  float(app_proto == "dcerpc"),
        float(app_proto == "smb"),  float(app_proto == "rdp"),
        float(app_proto == "failed"),
        float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state == "established"), float(state == "closed"),
        float(state == "new"),
        float(reason == "timeout"), float(reason == "rst"),
        float(reason == "fin"),
    ], dtype=np.float32)

    # 28 zenginlestirilmis ozellik
    enriched = np.zeros(28, dtype=np.float32)
    tcpf = _extract_tcp_flags(event)
    idx = 0
    enriched[idx:idx+5] = [tcpf["syn"], tcpf["ack"], tcpf["fin"], tcpf["rst"], tcpf["psh"]]
    idx += 5

    if enrichment:
        hf = _extract_http_features(enrichment)
        enriched[idx:idx+4] = hf["mo"];  idx += 4
        enriched[idx:idx+4] = hf["so"];  idx += 4
        enriched[idx] = hf["cth"]; idx += 1
        enriched[idx] = hf["cto"]; idx += 1
        enriched[idx] = hf["ctj"]; idx += 1
        enriched[idx] = hf["ct_"]; idx += 1

        tlsf = _extract_tls_features(enrichment)
        enriched[idx] = tlsf["e"];   idx += 1
        enriched[idx] = tlsf["sni"]; idx += 1

        dnsf = _extract_dns_features(enrichment)
        enriched[idx] = dnsf["e"]; idx += 1
        enriched[idx:idx+4] = [dnsf["qa"], dnsf["qaa"], dnsf["qm"], dnsf["qo"]];  idx += 4
        enriched[idx:idx+4] = [dnsf["rn"], dnsf["rnx"], dnsf["rr"], dnsf["ro"]];  idx += 4

    features = np.concatenate([base, enriched])
    return features


def load_eve_features(eve_path, max_samples=None, label_fn=None):
    """Load features from eve.json with optional labels."""
    import json
    X_list = []
    y_list = []
    count = 0
    
    with open(eve_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get('event_type') == 'flow':
                    feat = extract_features_v7_standalone(event)
                    if feat is not None:
                        X_list.append(feat)
                        if label_fn:
                            y_list.append(label_fn(event))
                        count += 1
            except json.JSONDecodeError:
                continue
    
    if not X_list:
        return np.array([]), np.array([]) if label_fn else None
    
    return np.stack(X_list).astype(np.float32), np.array(y_list) if y_list else None


def create_label_fn(attack_windows, victim_ips):
    from datetime import datetime
    windows = []
    for w in attack_windows:
        start = datetime.fromisoformat(w['start'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(w['end'].replace('Z', '+00:00'))
        windows.append((start, end))
    victim_set = set(victim_ips)
    
    def label_fn(event):
        ts_str = event.get('timestamp', '')
        if not ts_str:
            return 0
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except:
            return 0
        src_ip = event.get('src_ip', '')
        dest_ip = event.get('dest_ip', '')
        in_window = any(start <= ts <= end for start, end in windows)
        is_victim = src_ip in victim_set or dest_ip in victim_set
        return 1 if (in_window and is_victim) else 0
    return label_fn