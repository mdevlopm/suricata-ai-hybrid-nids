#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick validation: load model bundle, run single fake flow through pipeline."""

import pickle
import sys
sys.path.insert(0, '/run/media/mehmet/siber data1/ai modeli xgboost/pipeline')
import numpy as np
from pathlib import Path

# Import CORAL adapter from trainv8 (has fallback inline)
from trainv8 import CORALDomainAdapter

MODEL_PATH = Path("/run/media/mehmet/siber data1/ai modeli xgboost/models/ids_model_v8_final.pkl")

# Load bundle
print(f"Loading {MODEL_PATH}...")
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

bst = bundle['model']
scaler = bundle['scaler']
threshold = bundle['threshold']
coral = bundle['coral_adapter']
CLASS_MAP = bundle['class_map']
INV_CLASS_MAP = bundle['inv_class_map']

print(f"  Model: {type(bst).__name__}")
print(f"  Scaler: {type(scaler).__name__}")
print(f"  Threshold: {threshold:.4f}")
print(f"  CORAL fitted: {getattr(coral, 'is_fitted_', None)} ({'Enabled' if coral else 'Disabled - Raw XGBoost Mode'})")
print(f"  Classes: {CLASS_MAP}")

# Create a synthetic benign-like flow event (minimal valid flow)
fake_event = {
    "event_type": "flow",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "flow_id": "test_flow_123",
    "proto": "TCP",
    "app_proto": "http",
    "src_port": 54321,
    "dest_port": 80,
    "ip_v": 4,
    "flow": {
        "pkts_toserver": 10,
        "pkts_toclient": 8,
        "bytes_toserver": 1200,
        "bytes_toclient": 2400,
        "start": "2024-01-15T10:30:00.000Z",
        "end": "2024-01-15T10:30:02.500Z",
        "state": "established",
        "reason": "timeout",
        "age": 2.5
    }
}
fake_enrichment = {
    "http": {"http": {"http_method": "GET", "status": 200, "content_type": "text/html"}},
    "tls": {"tls": {"sni": "example.com"}},
    "dns": {"dns": {"queries": [{"rrtype": "A"}], "rcode": "NOERROR"}}
}

# Extract features (copy from trainv8)
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

def extract_http_features(enrichment):
    if not enrichment or "http" not in enrichment:
        return {"mo": [0,0,0,0], "so": [0,0,0,0], "cth": 0, "cto": 0, "ctj": 0, "ct_": 0}
    http = enrichment["http"].get("http", {})
    method = (http.get("http_method") or "").upper()
    status = http.get("status", 0) or 0
    ctype  = (http.get("content_type") or "").lower()
    mo = [float(method=="GET"), float(method=="POST"), float(method=="HEAD"), float(method not in ("GET","POST","HEAD"))]
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
        "e": 1, "qa": float("A" in qt), "qaa": float("AAAA" in qt),
        "qm": float("MX" in qt), "qo": float(bool(qt - {"A","AAAA","MX"})),
        "rn": float(rc == "NOERROR"), "rnx": float(rc == "NXDOMAIN"),
        "rr": float(rc == "REFUSED"), "ro": float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED")),
    }

def extract_features_v7(event, enrichment=None):
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

    enriched = np.zeros(28, dtype=np.float32)
    tcpf = _extract_tcp_flags(event)
    idx = 0
    enriched[idx:idx+5] = [tcpf["syn"], tcpf["ack"], tcpf["fin"], tcpf["rst"], tcpf["psh"]]
    idx += 5

    if enrichment:
        hf = extract_http_features(enrichment)
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

    return np.concatenate([base, enriched])

# Run pipeline
print("\n--- Pipeline Test ---")
feat = extract_features_v7(fake_event, fake_enrichment)
print(f"Raw features: shape={feat.shape}, dtype={feat.dtype}")

# Feature processing
if coral is not None:
    feat_aligned = coral.transform_target(feat.reshape(1, -1))
    feat_scaled = scaler.transform(feat_aligned)
    print(f"After CORAL+scaler: shape={feat_scaled.shape}")
else:
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    print(f"After scaler (Raw mode): shape={feat_scaled.shape}")

# XGBoost predict
if hasattr(bst, 'predict_proba'):
    probs = bst.predict_proba(feat_scaled)[0]
else:
    import xgboost as xgb
    dtest = xgb.DMatrix(feat_scaled)
    probs = bst.predict(dtest)[0]
pred_class = int(probs.argmax())
prob_benign = probs[0]
prob_attack = 1.0 - prob_benign
is_attack = prob_attack >= threshold

print(f"\nProbabilities:")
for i, p in enumerate(probs):
    print(f"  {INV_CLASS_MAP[i]}: {p:.4f}")
print(f"\nBinary: P(attack)={prob_attack:.4f}, threshold={threshold:.4f} -> {'ATTACK' if is_attack else 'BENIGN'}")
print(f"Multiclass prediction: {INV_CLASS_MAP[pred_class]}")

print("\n✅ Validation PASSED — model bundle loads and runs correctly")