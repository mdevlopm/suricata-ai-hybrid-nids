#!/usr/bin/env python3.11
"""RPi5 Benchmark: LSTM (v3_final_snapshot) + XGBoost v7 hybrid inference."""

import os, sys, json, pickle, time, gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import psutil

BASE = Path(__file__).parent.resolve()
V3_DIR = BASE / "model egitim dosyalari" / "v3_final_snapshot"
XGB_PATH = BASE / "ids_model_v7_final.pkl"
EVE_PATH = BASE / "model_json_dosyasi_test_bolumu" / "eve_test20.json"
WINDOW_SIZE = 40
XGB_DOS_DDOS_CLASSES = {1, 2}
LSTM_INV_CLASS_MAP = {0: "Volumetric", 1: "WebAttack", 2: "Bot"}


def _extract_tcp_flags(event):
    tcp = event.get("tcp") if event else None
    if not tcp:
        return {"syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0}
    try:
        flags_int = int(tcp.get("tcp_flags_ts", "00"), 16)
    except (ValueError, TypeError):
        flags_int = 0
    return {
        "syn": float(bool(flags_int & 0x02)),
        "ack": float(bool(flags_int & 0x10)),
        "fin": float(bool(flags_int & 0x01)),
        "rst": float(bool(flags_int & 0x04)),
        "psh": float(bool(flags_int & 0x08)),
    }


def _extract_http_features(enrichment):
    http_ev = enrichment.get("http") if enrichment else None
    if not http_ev:
        return {"mo": [0,0,0,0], "so": [0,0,0,0],
                "cth": 0, "cto": 0, "ctj": 0, "ct_": 0}
    h = http_ev.get("http", {})
    m = (h.get("http_method") or "").upper()
    mv = [0,0,0,0]
    if m == "GET":    mv[0] = 1
    elif m == "POST": mv[1] = 1
    elif m == "HEAD": mv[2] = 1
    elif m: mv[3] = 1
    s = int(h.get("status", 0) or 0)
    sv = [0,0,0,0]
    if 200 <= s < 300: sv[0] = 1
    elif 300 <= s < 400: sv[1] = 1
    elif 400 <= s < 500: sv[2] = 1
    elif s >= 500: sv[3] = 1
    ct = (h.get("http_content_type") or "").lower()
    return {
        "mo": mv, "so": sv,
        "cth": float("html" in ct),
        "cto": float("octet-stream" in ct or "binary" in ct),
        "ctj": float("json" in ct),
        "ct_": float(bool(ct) and not ("html" in ct or "octet-stream" in ct
                                         or "binary" in ct or "json" in ct)),
    }


def _extract_tls_features(enrichment):
    tls_ev = enrichment.get("tls") if enrichment else None
    if not tls_ev:
        return {"e": 0, "sni": 0}
    sni = (tls_ev.get("tls", {}).get("sni") or "")
    return {"e": 1, "sni": float(bool(sni))}


def _extract_dns_features(enrichment):
    dns_ev = enrichment.get("dns") if enrichment else None
    if not dns_ev:
        return {"e": 0, "qa": 0, "qaa": 0, "qm": 0, "qo": 0,
                "rn": 0, "rnx": 0, "rr": 0, "ro": 0}
    d = dns_ev.get("dns", {})
    qt = set((q.get("rrtype") or "").upper() for q in (d.get("queries") or []))
    rc = (d.get("rcode") or "").upper()
    return {
        "e": 1,
        "qa": float("A" in qt), "qaa": float("AAAA" in qt),
        "qm": float("MX" in qt), "qo": float(bool(qt - {"A","AAAA","MX"})),
        "rn": float(rc == "NOERROR"), "rnx": float(rc == "NXDOMAIN"),
        "rr": float(rc == "REFUSED"),
        "ro": float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED")),
    }


def extract_features_v7(event, enrichment):
    if event.get("event_type") != "flow":
        return None, None
    flow = event.get("flow", {})
    pts = float(flow.get("pkts_toserver", 0) or 0)
    ptc = float(flow.get("pkts_toclient", 0) or 0)
    bts = float(flow.get("bytes_toserver", 0) or 0)
    btc = float(flow.get("bytes_toclient", 0) or 0)
    try:
        from dateutil.parser import isoparse
        t0 = isoparse((flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1 = isoparse((flow.get("end") or "").replace("Z","+00:00"))
        dur = max((t1 - t0).total_seconds(), 0.0)
    except Exception:
        dur = 0.0
    dp = int(event.get("dest_port", 0) or 0)
    sp = int(event.get("src_port", 0) or 0)
    tp = pts + ptc
    tb = bts + btc
    sd = max(dur, 0.1)
    spk = max(tp, 1)
    sb = max(tb, 1)
    proto = (event.get("proto") or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v = int(event.get("ip_v", 4) or 4)
    state = (flow.get("state") or "").lower()
    reason = (flow.get("reason") or "").lower()
    age = float(flow.get("age", 0) or 0)
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
        hf = _extract_http_features(enrichment)
        enriched[idx:idx+4] = hf["mo"]; idx += 4
        enriched[idx:idx+4] = hf["so"]; idx += 4
        enriched[idx] = hf["cth"]; idx += 1
        enriched[idx] = hf["cto"]; idx += 1
        enriched[idx] = hf["ctj"]; idx += 1
        enriched[idx] = hf["ct_"]; idx += 1
        tlsf = _extract_tls_features(enrichment)
        enriched[idx] = tlsf["e"]; idx += 1
        enriched[idx] = tlsf["sni"]; idx += 1
        dnsf = _extract_dns_features(enrichment)
        enriched[idx] = dnsf["e"]; idx += 1
        enriched[idx:idx+4] = [dnsf["qa"], dnsf["qaa"], dnsf["qm"], dnsf["qo"]]
        idx += 4
        enriched[idx:idx+4] = [dnsf["rn"], dnsf["rnx"], dnsf["rr"], dnsf["ro"]]
        idx += 4
    features = np.concatenate([base, enriched]).reshape(1, -1)
    meta = {
        "timestamp": event.get("timestamp", ""),
        "src_ip": event.get("src_ip", ""),
        "dest_ip": event.get("dest_ip", ""),
        "src_port": sp, "dest_port": dp,
        "proto": proto, "app_proto": app_proto,
        "flow_id": event.get("flow_id", ""),
    }
    return features, meta


class FlowEnrichment:
    def __init__(self, max_entries=200_000):
        self._store = {}
        self._max = max_entries

    def ingest(self, event):
        etype = event.get("event_type")
        if etype not in ("http", "tls", "dns"):
            return
        fid = event.get("flow_id")
        if fid is None:
            return
        if len(self._store) >= self._max:
            return
        entry = self._store.setdefault(fid, {"_ts": datetime.now()})
        if etype not in entry:
            entry[etype] = event

    def get(self, flow_id):
        entry = self._store.pop(flow_id, None) if flow_id else None
        if entry:
            entry.pop("_ts", None)
            return entry
        return None

    def clear(self):
        self._store.clear()


class IPBuffer:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.buffers = {}
        self.last_seen = {}

    def add(self, src_ip, feature_vector, meta):
        if src_ip not in self.buffers:
            self.buffers[src_ip] = deque(maxlen=self.window_size)
        self.buffers[src_ip].append((feature_vector, meta))
        self.last_seen[src_ip] = datetime.now()

    def get_window(self, src_ip):
        if src_ip not in self.buffers:
            return None
        buf = self.buffers[src_ip]
        if len(buf) < self.window_size:
            return None
        features = np.array([item[0] for item in buf], dtype=np.float32)
        return features.reshape(1, self.window_size, -1).astype(np.float16)


def load_xgboost(path):
    print(f"  XGBoost model: {path}")
    with open(path, "rb") as f:
        b = pickle.load(f)
    model = b["model"]
    scaler = b["scaler"]
    thresh = b.get("threshold", 0.59)
    inv_cmap = b.get("inv_class_map", {})
    cmap = b.get("class_map", {})
    is_multiclass = len(cmap) > 2
    try:
        model.set_params(device="cpu")
    except Exception:
        pass
    return model, scaler, thresh, inv_cmap, cmap, is_multiclass


def load_lstm():
    model_path = V3_DIR / "lstm_best.keras"
    scaler_path = V3_DIR / "lstm_scaler.pkl"
    print(f"  LSTM model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def mem_usage():
    p = psutil.Process()
    return p.memory_info().rss / 1024 / 1024  # MB


def benchmark_lstm_only(lstm_model, lstm_scaler, n_warmup=10, n_bench=200):
    """Benchmark LSTM inference latency with synthetic data."""
    dummy_window = np.random.randn(1, WINDOW_SIZE, 78).astype(np.float16)
    # Warmup
    for _ in range(n_warmup):
        _ = lstm_model.predict(dummy_window, verbose=0)
    # Benchmark
    times = []
    mem_before = mem_usage()
    for _ in range(n_bench):
        t0 = time.perf_counter()
        window_norm = lstm_scaler.transform(dummy_window.reshape(-1, 78))
        window_norm = window_norm.reshape(1, WINDOW_SIZE, -1).astype(np.float16)
        prob = lstm_model.predict(window_norm, verbose=0)
        _ = int(np.argmax(prob[0]))
        times.append((time.perf_counter() - t0) * 1000)
    mem_after = mem_usage()
    return np.mean(times), np.std(times), mem_after - mem_before


def benchmark_xgb_only(xgb_model, xgb_scaler, dummy_features, n_warmup=100, n_bench=1000):
    """Benchmark XGBoost inference latency."""
    for _ in range(n_warmup):
        _ = xgb_model.predict_proba(xgb_scaler.transform(dummy_features))
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        xgb_prob = xgb_model.predict_proba(xgb_scaler.transform(dummy_features))[0]
        _ = int(xgb_prob.argmax())
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def benchmark_tflite(lstm_model, lstm_scaler, n_warmup=10, n_bench=200):
    """Convert to TFLite and benchmark."""
    print("\n  --- TFLite Conversion ---")
    t0 = time.time()
    converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    conv_time = time.time() - t0
    print(f"  Conversion time: {conv_time:.2f}s")
    print(f"  TFLite model size: {len(tflite_model) / 1024:.1f} KB")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    dummy = np.random.randn(1, WINDOW_SIZE, 78).astype(np.float16)
    window_norm = lstm_scaler.transform(dummy.reshape(-1, 78))
    window_norm = window_norm.reshape(1, WINDOW_SIZE, -1).astype(np.float16)
    tflite_input = window_norm.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], tflite_input)

    # Warmup
    for _ in range(n_warmup):
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        _ = int(np.argmax(output[0]))
        times.append((time.perf_counter() - t0) * 1000)

    tflite_mean = np.mean(times)
    tflite_std = np.std(times)
    return tflite_mean, tflite_std, len(tflite_model) / 1024


def run_end_to_end(xgb_model, xgb_scaler, xgb_threshold, inv_cmap, cmap, is_multiclass,
                    lstm_model, lstm_scaler):
    """End-to-end hybrid inference on real eve.json data."""
    print(f"\n  --- End-to-End Hybrid Inference ---")
    print(f"  Input: {EVE_PATH}")
    print(f"  XGB threshold: {xgb_threshold}")

    flow_cache = FlowEnrichment()
    ip_buffer = IPBuffer()

    stats = {"total": 0, "xgb_benign": 0, "xgb_dosddos": 0, "lstm_pending": 0, "lstm_triggered": 0}
    lstm_times = []
    xgb_times = []
    mem_before = mem_usage()
    t_start = time.time()

    with open(EVE_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue

            etype = event.get("event_type")
            if etype in ("http", "tls", "dns"):
                flow_cache.ingest(event)
                continue
            if etype != "flow":
                continue

            stats["total"] += 1
            enrichment = flow_cache.get(event.get("flow_id"))

            features, meta = extract_features_v7(event, enrichment)
            if features is None:
                continue

            # XGBoost
            t0 = time.perf_counter()
            xgb_prob = xgb_model.predict_proba(xgb_scaler.transform(features))[0]
            prob_atk = 1.0 - xgb_prob[0]
            xgb_pred = int(xgb_prob.argmax())
            xgb_times.append((time.perf_counter() - t0) * 1000)

            if prob_atk < xgb_threshold:
                stats["xgb_benign"] += 1
                continue
            if is_multiclass and xgb_pred == 0:
                stats["xgb_benign"] += 1
                continue

            if is_multiclass and xgb_pred in XGB_DOS_DDOS_CLASSES:
                stats["xgb_dosddos"] += 1
                continue

            src_ip = meta["src_ip"]
            flow_data = event.get("flow", {})
            bts = float(flow_data.get("bytes_toserver", 0) or 0)
            btc = float(flow_data.get("bytes_toclient", 0) or 0)
            ip_meta = {
                "ts": datetime.now(),
                "dest_ip": event.get("dest_ip", ""),
                "dest_port": int(event.get("dest_port", 0) or 0),
                "total_bytes": int(bts + btc),
                "dns_queries": [],
                "http_uri": None,
                "tls_sni": None,
            }
            if enrichment:
                dns_ev = enrichment.get("dns")
                if dns_ev:
                    d = dns_ev.get("dns", {})
                    ip_meta["dns_queries"] = [q.get("rrname", "") for q in (d.get("queries") or []) if q.get("rrname")]
                http_ev = enrichment.get("http")
                if http_ev:
                    ip_meta["http_uri"] = http_ev.get("http", {}).get("url", "")
                tls_ev = enrichment.get("tls")
                if tls_ev:
                    ip_meta["tls_sni"] = tls_ev.get("tls", {}).get("sni", "")

            ip_buffer.add(src_ip, features[0], ip_meta)
            stats["lstm_pending"] += 1

            window = ip_buffer.get_window(src_ip)
            if window is not None and lstm_model is not None:
                t0 = time.perf_counter()
                window_norm = lstm_scaler.transform(window.reshape(-1, window.shape[-1]))
                window_norm = window_norm.reshape(1, WINDOW_SIZE, -1).astype(np.float16)
                lstm_prob = lstm_model.predict(window_norm, verbose=0)[0]
                lstm_idx = int(np.argmax(lstm_prob))
                lstm_conf = float(lstm_prob[lstm_idx])
                lstm_times.append((time.perf_counter() - t0) * 1000)
                stats["lstm_triggered"] += 1

    elapsed = time.time() - t_start
    mem_after = mem_usage()
    ram_delta = mem_after - mem_before

    print(f"\n  Results:")
    print(f"  Total flows processed : {stats['total']:,}")
    print(f"  XGBoost Benign/skip   : {stats['xgb_benign']:,}")
    print(f"  XGBoost DoS/DDoS      : {stats['xgb_dosddos']:,}")
    print(f"  LSTM pending (buffer) : {stats['lstm_pending']:,}")
    print(f"  LSTM triggered        : {stats['lstm_triggered']:,}")
    print(f"  Elapsed time          : {elapsed:.2f}s")
    print(f"  Throughput            : {stats['total']/elapsed:,.0f} flows/s")

    if xgb_times:
        print(f"\n  XGBoost latency:")
        print(f"    Mean   : {np.mean(xgb_times):.3f} ms")
        print(f"    Std    : {np.std(xgb_times):.3f} ms")
        print(f"    Median : {np.median(xgb_times):.3f} ms")
        print(f"    P99    : {np.percentile(xgb_times, 99):.3f} ms")
    if lstm_times:
        print(f"\n  LSTM latency (per window):")
        print(f"    Mean   : {np.mean(lstm_times):.3f} ms")
        print(f"    Std    : {np.std(lstm_times):.3f} ms")
        print(f"    Median : {np.median(lstm_times):.3f} ms")
        print(f"    P99    : {np.percentile(lstm_times, 99):.3f} ms")
    print(f"\n  RAM usage delta: {ram_delta:.1f} MB")
    return stats, elapsed, ram_delta


def main():
    print("=" * 64)
    print("   RPi5 BENCHMARK: v3_final_snapshot LSTM + XGBoost v7")
    print("=" * 64)
    print(f"\n  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM")
    print(f"  TF: {tf.__version__}, GPU: {tf.config.list_physical_devices('GPU')}")

    # ── Load models ──
    print("\n  [1] Loading XGBoost v7...")
    xgb_model, xgb_scaler, xgb_thresh, inv_cmap, cmap, is_mc = load_xgboost(XGB_PATH)
    print(f"      Multiclass: {is_mc}, Threshold: {xgb_thresh}")

    print("\n  [2] Loading LSTM (v3_final_snapshot)...")
    lstm_model, lstm_scaler = load_lstm()
    params = sum(np.prod(w.shape) for w in lstm_model.weights)
    print(f"      Parameters: {params:,} ({params * 4 / 1024:.1f} KB fp32)")

    # ── Benchmarks ──
    print("\n  [3] Benchmark: XGBoost inference latency...")
    dummy_feat = np.random.randn(1, 70).astype(np.float32)
    xgb_mean, xgb_std = benchmark_xgb_only(xgb_model, xgb_scaler, dummy_feat)
    print(f"      Mean: {xgb_mean:.3f} ms, Std: {xgb_std:.3f} ms")

    print("\n  [4] Benchmark: LSTM inference latency (fp16, 40x78 window)...")
    lstm_mean, lstm_std, lstm_ram_delta = benchmark_lstm_only(lstm_model, lstm_scaler)
    print(f"      Mean: {lstm_mean:.3f} ms, Std: {lstm_std:.3f} ms")
    print(f"      RAM delta: {lstm_ram_delta:.1f} MB")

    print("\n  [5] TFLite conversion (fp16 quantized)...")
    tflite_mean, tflite_std, tflite_kb = benchmark_tflite(lstm_model, lstm_scaler)
    print(f"      Mean: {tflite_mean:.3f} ms, Std: {tflite_std:.3f} ms")

    print("\n  [6] End-to-end hybrid inference on test data...")
    stats, e2e_elapsed, e2e_ram = run_end_to_end(
        xgb_model, xgb_scaler, xgb_thresh, inv_cmap, cmap, is_mc,
        lstm_model, lstm_scaler
    )

    # ── Final Report ──
    print("\n" + "=" * 64)
    print("   FINAL REPORT: RPi5 DEPLOYMENT ANALYSIS")
    print("=" * 64)

    lstm_model_size_mb = params * 4 / 1024 / 1024
    speedup = lstm_mean / tflite_mean if tflite_mean > 0 else 0
    print(f"""
  LSTM Model:
    Architecture : BiLSTM(64)+LSTM(32)+Dense(32) -> 3 classes
    Input shape  : (40, 78)  [40 timesteps x 78 features]
    Parameters   : {params:,} ({lstm_model_size_mb:.2f} MB fp32)
    Latency (Keras fp16) : {lstm_mean:.2f} ms
    Latency (TFLite fp16): {tflite_mean:.2f} ms
    Speedup with TFLite  : {speedup:.2f}x

  XGBoost v7:
    Features     : 70
    Classes      : 6
    Trees        : 1500
    Latency      : {xgb_mean:.3f} ms/flow

  End-to-End Pipeline:
    Throughput   : {stats['total']/e2e_elapsed:,.0f} flows/s
    RAM delta    : {e2e_ram:.1f} MB
    LSTM hit rate: {stats['lstm_triggered']/max(stats['total'],1)*100:.2f}%

  TFLite/ONNX Conversion Recommendation:
    - Keras latency  : {lstm_mean:.2f} ms  ({1/(lstm_mean/1000):.0f} inferences/s)
    - TFLite latency : {tflite_mean:.2f} ms  ({1/(tflite_mean/1000):.0f} inferences/s)
""")
    # Decision logic
    if lstm_mean < 10:
        print("  >> KADAR: Keras formati yeterli (< 10 ms). TFLite'a gecmeye GEREK YOK.")
    elif speedup > 1.5:
        print(f"  >> GEREKLI: Keras {lstm_mean:.1f}ms -> TFLite {tflite_mean:.1f}ms ({speedup:.1f}x hizli).")
    else:
        print(f"  >> GEREKLI DEGIL: Keras {lstm_mean:.1f}ms yeterli hizda.")
    print("=" * 64)


if __name__ == "__main__":
    main()