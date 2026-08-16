# -*- coding: utf-8 -*-
"""
lstm_train.py - LSTM IDS Egitim Scripti
=========================================
implementation.md §5-6'ya tam uyumlu.
CICIDS2018 + CTU-13 + MCFP EVE JSON'larindan kayan penceler olusturur,
LSTM modeli egitir ve hibrit cikarim icin kaydeder.

Ciktilari (model eğitim dosyalari/ klasorune):
  - lstm_best.keras      - LSTM model (TensorFlow SavedModel)
  - lstm_scaler.pkl       - StandardScaler
  - lstm_metadata.pkl     - Egittim metadatasi
  - (gecici) lstm_chunks/ - Ara parcalar

Calistirma:
  python3 lstm_train.py
"""

import gc, json, pickle, shutil, time, hashlib, warnings, os, psutil
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict, deque

from features import compute_ip_window_features

def mem(label=""):
    p = psutil.Process()
    gb = p.memory_info().rss / 1024**3
    print(f"  [MEM {label}] {gb:.2f} GB")
    return gb

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf

# GPU bellek yonetimi: 5.5GB VRAM siniri
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=5120)])
    except:
        pass

from tensorflow.keras import layers, models, mixed_precision

warnings.filterwarnings("ignore")
mixed_precision.set_global_policy("mixed_float16")

# ── AYARLAR ─────────────────────────────────────────────────────────────────
BASE_DIR = Path("/run/media/mehmet/siber data1/ai modeli xgboost/pcap dosyaları ve veri setleri")
OUT_DIR  = Path(__file__).parent.resolve()

WINDOW_SIZE = 40
STRIDE      = 1
GAP_SEC     = 300.0
CHUNK_SIZE  = 5_000          # 5K pencere/chunk → 12GB RAM guvenli
VAL_SPLIT   = 0.20
RANDOM_SEED = 42
MAX_SEQUENCES_TOTAL = 150_000  # 50K/sinif × 3 sinif, GPU 5.5GB VRAM guvenli

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── SINIF TANIMLARI (3 sinif: Volumetric, WebAttack, Bot) ─────────────────────
# Volumetric = DoS + DDoS + Infiltration (flow-stat uzayinda ayrismiyorlar,
#   literatur ve deneylerle dogrulandi: Liu & Engelen 2022)
# DoS/DDoS LSTM cikisinda "Volumetric" olarak etiketlenir.
# XGBoost multiclass ayri bir katmanda DoS/DDoS'u zaten tanir.
CLASS_MAP = {
    "Benign":       0,
    "DoS": 1, "DDoS": 1, "Infiltration": 1,  # → Volumetric
    "WebAttack":    2,
    "Bot":          3,
}
INV_CLASS_MAP = {1: "Volumetric", 2: "WebAttack", 3: "Bot"}
ATTACK_IDS    = sorted(set(c for c in CLASS_MAP.values() if c != 0))  # [1,2,3]

# LSTM 3 sinif: 1->0, 2->1, 3->2
LSTM_CLASSES = len(ATTACK_IDS)

# ── VERI KAYNAKLARI ─────────────────────────────────────────────────────────
# Volumetric = DoS + DDoS + Infiltration (3 kaynak birlikte)
# DoS/DDoS LSTM'de "Volumetric" sinifina dahil edildi
EVE_FILES = {
    "Benign": [
        BASE_DIR / "Wednesday-14-02-2018/eve_Benign.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407021400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407071400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408111400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408121400.json",
    ],
    "DoS": [
        BASE_DIR / "Thursday-15-02-2018/eve_DoS.json",
        BASE_DIR / "Friday-16-02-2018/eve_DoS.json",
    ],
    "DDoS": [
        BASE_DIR / "Tuesday-20-02-2018/eve_DDoS.json",
        BASE_DIR / "Wednesday-21-02-2018/eve_DDoS.json",
    ],
    "Infiltration": [
        BASE_DIR / "Wednesday-28-02-2018/eve_Infiltration.json",
        BASE_DIR / "Thursday-01-03-2018/eve_Infiltration.json",
    ],
    "WebAttack": [
        BASE_DIR / "Thursday-22-02-2018/eve_WebAttack.json",
        BASE_DIR / "Friday-23-02-2018/eve_WebAttack.json",
    ],
    "Bot": [
        BASE_DIR / "Friday-02-03-2018/eve_Bot.json",
        BASE_DIR / "Ctu-13/eve_botnet_neris.json",
        BASE_DIR / "mcfp felk/eve_botnet_mcfp.json",
    ],
}

# ── OZELLIK TANIMI (v7 enriched: 42 base + 28 yeni) ──────────────────────
FEATURE_COLS_BASE = [
    "duration_s", "pkts_toserver", "pkts_toclient",
    "bytes_toserver", "bytes_toclient", "total_pkts", "total_bytes",
    "pkt_rate", "byte_rate", "bytes_per_pkt",
    "upload_byte_ratio", "upload_pkt_ratio", "byte_asymmetry",
    "download_byte_ratio", "download_pkt_ratio",
    "flow_age", "is_unidirectional", "pkt_size_variance_proxy",
    "dest_port", "src_port",
    "is_well_known_port", "is_registered_port", "is_high_port",
    "is_same_port", "is_ipv6", "is_tcp", "is_udp", "is_icmp",
    "app_http", "app_dns", "app_tls",
    "app_dcerpc", "app_smb", "app_rdp",
    "app_failed", "app_unknown",
    "state_established", "state_closed", "state_new",
    "reason_timeout", "reason_rst", "reason_fin",
]
FEATURE_COLS = FEATURE_COLS_BASE + [
    "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh",
    "http_method_get", "http_method_post", "http_method_head", "http_method_other",
    "http_status_2xx", "http_status_3xx", "http_status_4xx", "http_status_5xx",
    "http_content_type_html", "http_content_type_octet", "http_content_type_json", "http_content_type_other",
    "tls_exists", "tls_sni_exists",
    "dns_exists", "dns_query_a", "dns_query_aaaa", "dns_query_mx", "dns_query_other",
    "dns_rcode_noerror", "dns_rcode_nxdomain", "dns_rcode_refused", "dns_rcode_other",
    # IP-window behavioral features (Bot vs WebAttack ayrimi icin)
    "beacon_interval_mean", "beacon_interval_std",
    "dst_ip_entropy", "dns_per_min",
    "uri_entropy", "same_dst_port_ratio",
    "tls_sni_reuse", "payload_size_variance",
]
N_FEATURES_v7 = len(FEATURE_COLS)
N_FEATURES_v7_CORE = 70  # v7 core without IP-window features
N_FEATURES_BASE = len(FEATURE_COLS_BASE)


class FlowEnrichment:
    """Single-pass cache for http/tls/dns events keyed by flow_id."""
    def __init__(self, max_entries=500_000):
        self._store = {}; self._max = max_entries
    def ingest(self, event: dict):
        etype = event.get("event_type")
        if etype not in ("http", "tls", "dns"): return
        fid = event.get("flow_id")
        if fid is None or len(self._store) >= self._max: return
        entry = self._store.setdefault(fid, {})
        if etype not in entry: entry[etype] = event
    def get(self, flow_id):
        return self._store.pop(flow_id, None) if flow_id else None


def extract_tcp_flags(flow_event: dict) -> dict:
    tcp = flow_event.get("tcp") if flow_event else None
    if not tcp: return {"syn":0,"ack":0,"fin":0,"rst":0,"psh":0}
    try: flags_int = int(tcp.get("tcp_flags_ts","00"), 16)
    except: flags_int = 0
    return {"syn":float(bool(flags_int&2)),"ack":float(bool(flags_int&16)),
            "fin":float(bool(flags_int&1)),"rst":float(bool(flags_int&4)),
            "psh":float(bool(flags_int&8))}

def extract_http_features(http_event: dict) -> dict:
    if not http_event: return {"mo":[0,0,0,0],"so":[0,0,0,0],"cth":0,"cto":0,"ctj":0,"ct_":0}
    h = http_event.get("http",{})
    m = (h.get("http_method") or "").upper()
    mv = [0,0,0,0]
    if m=="GET": mv[0]=1
    elif m=="POST": mv[1]=1
    elif m=="HEAD": mv[2]=1
    elif m: mv[3]=1
    s = int(h.get("status",0) or 0)
    sv = [0,0,0,0]
    if 200<=s<300: sv[0]=1
    elif 300<=s<400: sv[1]=1
    elif 400<=s<500: sv[2]=1
    elif s>=500: sv[3]=1
    ct = (h.get("http_content_type") or "").lower()
    return {"mo":mv,"so":sv,"cth":float("html" in ct),"cto":float("octet-stream" in ct or "binary" in ct),
            "ctj":float("json" in ct),"ct_":float(bool(ct) and not ("html" in ct or "octet-stream" in ct or "binary" in ct or "json" in ct))}

def extract_tls_features(tls_event: dict) -> dict:
    if not tls_event: return {"e":0,"sni":0}
    sni = (tls_event.get("tls",{}).get("sni") or "")
    return {"e":1,"sni":float(bool(sni))}

def extract_dns_features(dns_event: dict) -> dict:
    if not dns_event: return {"e":0,"qa":0,"qaa":0,"qm":0,"qo":0,"rn":0,"rnx":0,"rr":0,"ro":0}
    d = dns_event.get("dns",{})
    qt = set((q.get("rrtype") or "").upper() for q in (d.get("queries") or []))
    rc = (d.get("rcode") or "").upper()
    return {"e":1,"qa":float("A" in qt),"qaa":float("AAAA" in qt),"qm":float("MX" in qt),"qo":float(bool(qt-{"A","AAAA","MX"})),
            "rn":float(rc=="NOERROR"),"rnx":float(rc=="NXDOMAIN"),"rr":float(rc=="REFUSED"),"ro":float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED"))}

def extract_features_v7(event: dict, enrichment: dict) -> np.ndarray:
    if event.get("event_type") != "flow": return None
    flow = event.get("flow", {})
    pts=float(flow.get("pkts_toserver",0) or 0); ptc=float(flow.get("pkts_toclient",0) or 0)
    bts=float(flow.get("bytes_toserver",0) or 0); btc=float(flow.get("bytes_toclient",0) or 0)
    try:
        from dateutil.parser import isoparse
        t0=isoparse((flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1=isoparse((flow.get("end") or "").replace("Z","+00:00"))
        dur=max((t1-t0).total_seconds(),0.0)
    except: dur=0.0
    dp=int(event.get("dest_port",0) or 0); sp=int(event.get("src_port",0) or 0)
    tp=pts+ptc; tb=bts+btc
    MIN_DUR=0.1; sd=max(dur,MIN_DUR); spk=max(tp,1); sb=max(tb,1)
    proto=(event.get("proto") or "").upper(); app_proto=(event.get("app_proto") or "unknown").lower()
    ip_v=int(event.get("ip_v",4) or 4); state=(flow.get("state") or "").lower()
    reason=(flow.get("reason") or "").lower(); age=float(flow.get("age",0) or 0)
    ts_str=event.get("timestamp","")
    try: ts=isoparse(ts_str.replace("Z","+00:00"))
    except: ts=datetime.min
    base=np.array([dur,pts,ptc,bts,btc,tp,tb,tp/sd,tb/sd,tb/spk,bts/sb,pts/spk,abs(bts-btc)/sb,btc/sb,ptc/spk,age,
        float(ptc==0),abs((bts/max(pts,1))-(btc/max(ptc,1))),dp,sp,float(dp<1024),float(1024<=dp<49152),float(dp>=49152),float(sp==dp),
        float(ip_v==6),float(proto=="TCP"),float(proto=="UDP"),float(proto in ("ICMP","ICMPv6")),
        float(app_proto=="http"),float(app_proto=="dns"),float(app_proto=="tls"),float(app_proto=="dcerpc"),float(app_proto=="smb"),float(app_proto=="rdp"),
        float(app_proto=="failed"),float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state=="established"),float(state=="closed"),float(state=="new"),float(reason=="timeout"),float(reason=="rst"),float(reason=="fin")],dtype=np.float32)
    enriched=np.zeros(N_FEATURES_v7_CORE-N_FEATURES_BASE,dtype=np.float32)
    tcpf=extract_tcp_flags(event); idx=0
    enriched[idx:idx+5]=[tcpf["syn"],tcpf["ack"],tcpf["fin"],tcpf["rst"],tcpf["psh"]]; idx+=5
    if enrichment:
        hf=extract_http_features(enrichment)
        enriched[idx:idx+4]=hf["mo"]; idx+=4
        enriched[idx:idx+4]=hf["so"]; idx+=4
        enriched[idx]=hf["cth"]; idx+=1; enriched[idx]=hf["cto"]; idx+=1; enriched[idx]=hf["ctj"]; idx+=1; enriched[idx]=hf["ct_"]; idx+=1
        tlsf=extract_tls_features(enrichment); enriched[idx]=tlsf["e"]; idx+=1; enriched[idx]=tlsf["sni"]; idx+=1
        dnsf=extract_dns_features(enrichment); enriched[idx]=dnsf["e"]; idx+=1
        enriched[idx:idx+4]=[dnsf["qa"],dnsf["qaa"],dnsf["qm"],dnsf["qo"]]; idx+=4
        enriched[idx:idx+4]=[dnsf["rn"],dnsf["rnx"],dnsf["rr"],dnsf["ro"]]; idx+=4
    return np.concatenate([base,enriched]), ts


# ── IP-WINDOW METADATA CIKARIMI ─────────────────────────────────────────────
def _extract_dns_queries(enrichment):
    if not enrichment: return []
    dns_ev = enrichment.get("dns")
    if not dns_ev: return []
    dns = dns_ev.get("dns", {})
    return [q.get("rrname", "") for q in (dns.get("queries") or []) if q.get("rrname")]

def _extract_http_uri(enrichment):
    if not enrichment: return None
    http_ev = enrichment.get("http")
    if not http_ev: return None
    return http_ev.get("http", {}).get("url", "")

def _extract_tls_sni(enrichment):
    if not enrichment: return None
    tls_ev = enrichment.get("tls")
    if not tls_ev: return None
    return tls_ev.get("tls", {}).get("sni", "")


# ── VERI YUKLEME ve GRUPLAMA (v7 enriched single-pass) ────────────────────
def load_flows(path: Path, label_id: int, max_flows: int = 200_000):
    if not path.exists(): return []
    cache = FlowEnrichment(max_entries=max_flows * 2)
    flows, count = [], 0
    fname_key = path.stem
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if count >= max_flows: break
            line = line.strip()
            if not line: continue
            try: ev = json.loads(line)
            except: continue
            etype = ev.get("event_type")
            if etype == "flow":
                enrichment = cache.get(ev.get("flow_id"))
                feat_ts = extract_features_v7(ev, enrichment)
                if feat_ts is not None:
                    feat, ts = feat_ts
                    src_ip = ev.get("src_ip", "0.0.0.0")
                    flow_data = ev.get("flow", {})
                    bts = float(flow_data.get("bytes_toserver", 0) or 0)
                    btc = float(flow_data.get("bytes_toclient", 0) or 0)
                    meta = {
                        "ts": ts,
                        "dest_ip": ev.get("dest_ip", ""),
                        "dest_port": int(ev.get("dest_port", 0) or 0),
                        "total_bytes": int(bts + btc),
                        "dns_queries": _extract_dns_queries(enrichment),
                        "http_uri": _extract_http_uri(enrichment),
                        "tls_sni": _extract_tls_sni(enrichment),
                    }
                    flows.append((fname_key, src_ip, ts, feat, label_id, meta))
                    count += 1
            elif etype in ("http", "tls", "dns"):
                cache.ingest(ev)
    return flows


def group_and_sort(flows):
    groups = defaultdict(list)
    for fname, src_ip, ts, feat, label, meta in flows:
        gid = (fname, src_ip)
        groups[gid].append((ts, feat, label, meta))
    for gid in groups:
        groups[gid].sort(key=lambda x: x[0])
    return groups


def split_by_gap(group, gap_sec=GAP_SEC):
    items = group
    if not items:
        return []
    sessions = []
    cur = [items[0]]
    for i in range(1, len(items)):
        gap = (items[i][0] - items[i-1][0]).total_seconds()
        if gap > gap_sec:
            sessions.append(cur)
            cur = []
        cur.append(items[i])
    if cur:
        sessions.append(cur)
    return sessions


def make_windows(sessions, group_id):
    windows_X, windows_y, windows_g = [], [], []
    for sess in sessions:
        n = len(sess)
        if n < WINDOW_SIZE:
            continue
        for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
            win = sess[i:i+WINDOW_SIZE]
            labels = [item[2] for item in win]
            if all(l == 0 for l in labels):
                continue
            # IP-window behavioral features (Bot vs WebAttack ayrimi)
            meta_list = [item[3] for item in win]
            win_features = compute_ip_window_features(meta_list)
            # Append window features to each flow's feature vector
            X_seq = np.stack([
                np.concatenate([item[1], win_features])
                for item in win
            ])
            # Majority vote (1-5), tie-break en kucuk index
            counter = Counter(l for l in labels if l != 0)
            if not counter:
                continue
            majority = counter.most_common()
            best_label = majority[0][0]
            if len(majority) > 1 and majority[0][1] == majority[1][1]:
                best_label = min(majority[0][0], majority[1][0])
            windows_X.append(X_seq)
            windows_y.append(best_label)
            windows_g.append(group_id)
    return windows_X, windows_y, windows_g


def prepare_data():
    """Ana veri hazirlik pipeline'i:
       1) Tum EVE JSON'lardan flow'lari oku
       2) (dosya_adi, src_ip) bazinda grupla
        3) 300sn boslukta oturumlara bol
        4) 40-akislik kayan pencere (stride=1) olustur
       5) Chunk'lara bolerek diske yaz
       6) Geri don
    """
    print("=" * 64)
    print("   LSTM VERI HAZIRLIGI")
    print("=" * 64)

    chunk_dir = OUT_DIR / "lstm_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir()

    all_flows = []
    for label_name, paths in EVE_FILES.items():
        label_id = CLASS_MAP[label_name]
        path_list = paths if isinstance(paths, list) else [paths]
        for p in path_list:
            flows = load_flows(p, label_id)
            print(f"  {label_name:<15} {p.name:<50} -> {len(flows):>8,} flow")
            all_flows.extend(flows)
            del flows
            gc.collect()

    print(f"\n  Toplam flow: {len(all_flows):,}")

    # Grupla
    print("  Gruplaniyor...")
    grouped = group_and_sort(all_flows)
    del all_flows
    gc.collect()
    print(f"  Grup sayisi: {len(grouped):,}")

    # Her grubun dominant sinifini bul
    rng = np.random.default_rng(RANDOM_SEED)
    group_items = list(grouped.items())
    rng.shuffle(group_items)

    group_id_map = {}
    next_gid = 0
    for gid, _ in group_items:
        group_id_map[gid] = next_gid
        next_gid += 1

    # Balanced sampling: her siniftan esit sayida pencere
    print("  Pencereler olusturuluyor...")
    chunk_buf_X, chunk_buf_y, chunk_buf_g = [], [], []
    chunk_idx = 0
    total_windows = 0

    # Once her gruptaki ilk tuple'dan dominant label'i tahmin et
    from collections import defaultdict as ddict
    per_class_windows = ddict(int)
    max_per_class = MAX_SEQUENCES_TOTAL // len(ATTACK_IDS)

    for gid_key, group_flows in group_items:
        grp_id = group_id_map[gid_key]
        sessions = split_by_gap(group_flows)
        wx, wy, wg = make_windows(sessions, grp_id)
        if not wx:
            continue

        # Bu grup icin dominant label
        dominant = Counter(wy).most_common(1)[0][0]
        if per_class_windows[dominant] >= max_per_class:
            continue

        # Kac tane ekleyebiliriz?
        take = min(len(wx), max_per_class - per_class_windows[dominant])
        wx = wx[:take]
        wy = wy[:take]
        wg = wg[:take]

        chunk_buf_X.extend(wx)
        chunk_buf_y.extend(wy)
        chunk_buf_g.extend(wg)
        total_windows += len(wx)
        per_class_windows[dominant] += len(wx)

        if len(chunk_buf_X) >= CHUNK_SIZE:
            _flush_chunk(chunk_dir, chunk_idx, chunk_buf_X, chunk_buf_y, chunk_buf_g)
            chunk_idx += 1
            chunk_buf_X, chunk_buf_y, chunk_buf_g = [], [], []
            gc.collect()

        if all(per_class_windows[c] >= max_per_class for c in ATTACK_IDS):
            break

    # Kalan
    if chunk_buf_X:
        _flush_chunk(chunk_dir, chunk_idx, chunk_buf_X, chunk_buf_y, chunk_buf_g)
        chunk_idx += 1
        del chunk_buf_X, chunk_buf_y, chunk_buf_g
        gc.collect()

    del grouped
    gc.collect()

    print(f"  Toplam pencere: {total_windows:,}")
    print(f"  Sinif bazli dagilim:")
    for c in ATTACK_IDS:
        cnt = per_class_windows.get(c, 0)
        name = INV_CLASS_MAP[c]
        print(f"    {name:<15} (label {c}): {cnt:>8,}")
    print(f"  Chunk sayisi  : {chunk_idx}")
    return chunk_dir, chunk_idx, group_id_map


def _flush_chunk(chunk_dir, idx, X, y, g):
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int8)
    g_arr = np.asarray(g, dtype=np.int32)
    np.save(chunk_dir / f"X_chunk_{idx:04d}.npy", X_arr)
    np.save(chunk_dir / f"y_chunk_{idx:04d}.npy", y_arr)
    np.save(chunk_dir / f"groups_chunk_{idx:04d}.npy", g_arr)
    print(f"    chunk {idx:04d} -> X:{X_arr.shape} y:{y_arr.shape} g:{g_arr.shape}")


def load_chunks(chunk_dir, chunk_count):
    X_list, y_list, g_list = [], [], []
    for i in range(chunk_count):
        X_list.append(np.load(chunk_dir / f"X_chunk_{i:04d}.npy"))
        y_list.append(np.load(chunk_dir / f"y_chunk_{i:04d}.npy"))
        g_list.append(np.load(chunk_dir / f"groups_chunk_{i:04d}.npy"))
    X = np.concatenate(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int8)
    g = np.concatenate(g_list).astype(np.int32)
    return X, y, g


# ── STRATIFIED GROUP SPLIT ─────────────────────────────────────────────────
def stratified_group_split(X, y, g, val_split=VAL_SPLIT, random_seed=RANDOM_SEED):
    unique_groups = np.unique(g)
    group_dominant = {}
    for grp in unique_groups:
        mask = g == grp
        labels_in_group = y[mask]
        counter = Counter(labels_in_group.tolist())
        group_dominant[grp] = counter.most_common(1)[0][0]
    group_ids = np.array(sorted(unique_groups))
    group_dom = np.array([group_dominant[grp] for grp in group_ids])

    rng = np.random.default_rng(random_seed)
    val_gids = set()
    present_classes = np.unique(group_dom)
    for c in present_classes:
        c_mask = group_dom == c
        c_group_ids = group_ids[c_mask]
        c_perm = rng.permutation(len(c_group_ids))
        n_val = max(1, int(len(c_group_ids) * val_split))
        val_gids.update(c_group_ids[c_perm[:n_val]])
    train_gids = set(group_ids) - val_gids
    train_mask = np.isin(g, list(train_gids))
    val_mask = np.isin(g, list(val_gids))
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask], g[train_mask], g[val_mask]


# ── LSTM MODELI ────────────────────────────────────────────────────────────
def build_lstm(input_shape=(WINDOW_SIZE, len(FEATURE_COLS)), num_classes=LSTM_CLASSES):
    l2_reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape, dtype=tf.float16)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.LSTM(32, return_sequences=False, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2_reg)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype=tf.float32)(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ids_lstm")
    return model


# ── ANA EGITIM ─────────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 64)
    print("   LSTM IDS MODEL EGITIMI")
    print("=" * 64)
    print(f"  GPU        : {tf.config.list_physical_devices('GPU')}")
    print(f"  Windows    : {WINDOW_SIZE}")
    print(f"  Ozellik    : {len(FEATURE_COLS)}")
    print(f"  Sinif      : {LSTM_CLASSES} -> {list(INV_CLASS_MAP.values())}")
    print(f"  Mixed prec : {mixed_precision.global_policy().name}")
    print()

    # --- 1) Veri hazirlik ---
    t0 = time.time()
    chunk_dir, chunk_count, group_id_map = prepare_data()
    print(f"\n  Veri hazirlama: {time.time() - t0:.1f} sn")

    # --- 2) Chunk'lari yukle ---
    t0 = time.time()
    print("  Chunk'lar birlestiriliyor...")
    X, y, g = load_chunks(chunk_dir, chunk_count)
    mem("chunks-loaded")
    if mem() > 8:
        print("  HATA: Bellek 8GB ustu, durduruluyor (12GB RAM siniri)!")
        del X, y, g; gc.collect(); shutil.rmtree(chunk_dir); return
    print(f"  X: {X.shape}  y: {y.shape}  g: {g.shape}")
    print(f"  Yukleme suresi: {time.time() - t0:.1f} sn")

    # Orijinal etiket 1-5 -> 0-4
    y_shifted = y - 1

    # --- 3) Stratified Group Split (group-aware, no leakage) ---
    t0 = time.time()
    print("  Stratified group split (80/20)...")
    result = stratified_group_split(X, y_shifted, g, val_split=VAL_SPLIT)
    if result is None:
        print("  HATA: Group split basarisiz!")
        return
    X_tr, X_val, y_tr, y_val, g_tr, g_val = result
    del X, y, g
    gc.collect()
    mem("after-split")
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}")
    print(f"  Split suresi: {time.time() - t0:.1f} sn")

    # --- 4) Normalizasyon ---
    t0 = time.time()
    print("  StandardScaler fit ediliyor...")
    orig_shape = X_tr.shape
    X_tr_2d = X_tr.reshape(-1, orig_shape[-1])
    scaler = StandardScaler()
    scaler.fit(X_tr_2d)
    del X_tr_2d
    gc.collect()
    X_tr = scaler.transform(X_tr.reshape(-1, orig_shape[-1])).reshape(orig_shape).astype(np.float16)
    X_val = scaler.transform(X_val.reshape(-1, orig_shape[-1])).reshape(X_val.shape).astype(np.float16)
    gc.collect()
    mem("after-norm")
    print(f"  Normalizasyon: {time.time() - t0:.1f} sn")

    # --- 5) Sinif agirliklari ---
    print("  Sinif agirliklari hesaplaniyor...")
    all_classes = np.arange(LSTM_CLASSES)
    present_classes = np.unique(y_tr)
    cw_dict = {}
    if len(present_classes) == LSTM_CLASSES:
        class_weights = compute_class_weight("balanced", classes=all_classes, y=y_tr)
        cw_dict = {int(c): float(w) for c, w in zip(all_classes, class_weights)}
    else:
        present_weights = compute_class_weight("balanced", classes=present_classes, y=y_tr)
        for c in all_classes:
            if c in present_classes:
                idx = np.where(present_classes == c)[0][0]
                cw_dict[int(c)] = float(present_weights[idx])
            else:
                cw_dict[int(c)] = 1.0
    print(f"  Class weights: {cw_dict}")

    # --- 6) Model ---
    print("  LSTM modeli olusturuluyor...")
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        model = build_lstm()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy",
                     tf.keras.metrics.SparseCategoricalCrossentropy(name="ce_loss")],
        )
    model.summary()

    # --- 7) Egitim ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_DIR / "lstm_best.keras"), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print(f"\n  Egitim basliyor... (batch=256, epoch=50, 5.5GB VRAM)")
    print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}")
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - t0
    best_epoch = int(np.argmin(history.history["val_loss"]))

    # --- 8) Degerlendirme ---
    print(f"\n{'=' * 64}")
    print("   LSTM TEST SONUCLARI")
    print(f"{'=' * 64}")
    print(f"  Egitim suresi : {train_time / 60:.1f} dk")
    print(f"  En iyi epoch  : {best_epoch + 1}")

    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_val_true = y_val.astype(int)

    y_val_true_orig = y_val_true + 1
    y_val_pred_orig = y_val_pred + 1

    present_labels = sorted(np.unique(y_val_true_orig).tolist())
    present_names  = [INV_CLASS_MAP[i] for i in present_labels]

    print(f"\n  Sinif Bazli Rapor:")
    print(classification_report(
        y_val_true_orig, y_val_pred_orig,
        labels=present_labels, target_names=present_names, zero_division=0
    ))

    macro_f1 = f1_score(y_val_true_orig, y_val_pred_orig, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val_true_orig, y_val_pred_orig, average="weighted", zero_division=0)
    print(f"\n  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    cm = confusion_matrix(y_val_true_orig, y_val_pred_orig, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)
    print(f"\n  Karisiklik Matrisi:")
    print(cm_df.to_string())

    # --- 9) Kaydet ---
    print(f"\n  Kaydediliyor...")
    scaler_path = OUT_DIR / "lstm_scaler.pkl"
    meta_path   = OUT_DIR / "lstm_metadata.pkl"
    model_path  = OUT_DIR / "lstm_best.keras"

    metadata = {
        "window_size"      : WINDOW_SIZE,
        "n_features"       : len(FEATURE_COLS),
        "feature_cols"     : FEATURE_COLS,
        "lstm_classes"     : LSTM_CLASSES,
        "attack_class_map" : {k: v for k, v in CLASS_MAP.items() if v != 0},
        "class_map"        : CLASS_MAP,
        "inv_class_map"    : INV_CLASS_MAP,
        "trained_on"       : "CICIDS2018 + CTU-13 + MCFP (LSTM 3-class: Volumetric/WebAttack/Bot, Volumetric=DoS+DDoS+Infiltration)",
        "macro_f1"         : round(float(macro_f1), 4),
        "weighted_f1"      : round(float(weighted_f1), 4),
        "training_date"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch"       : int(best_epoch + 1),
        "training_min"     : round(train_time / 60, 2),
        "n_train"          : len(y_tr),
        "n_val"            : len(y_val),
    }
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"  Model    : {model_path}")
    print(f"  Scaler   : {scaler_path}")
    print(f"  Metadata : {meta_path}")
    print(f"  Macro F1 : {macro_f1 * 100:.2f}%")
    shutil.rmtree(chunk_dir)
    print(f"\n  EGITIM TAMAMLANDI")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    import os
    train()
