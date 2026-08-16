#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ip_buffer.py - IP-Based Flow Buffering and LSTM Alarm Generation Module
================================================================-------
Gereksinimler:
- Pencere: 40 akış, 300sn gap (zaman farkı) ile flush
- Gruplama: src_ip bazlı
- LSTM threshold mantığı: Olasılık < threshold (varsayılan 0.50) → "Generic Attack" etiketi
- Çıktı: Alarm JSON şeması — {src_ip, class, confidence, window_start, window_end}
"""

import json
from collections import deque
from datetime import datetime, timezone
import numpy as np

try:
    from dateutil.parser import isoparse
except ImportError:
    def isoparse(ts_str):
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))

try:
    from features import compute_ip_window_features
except ImportError:
    try:
        from model_eğitim_dosyaları.features import compute_ip_window_features
    except ImportError:
        def compute_ip_window_features(meta_list):
            return np.zeros(8, dtype=np.float32)


def _parse_timestamp(ts):
    """Timestamp değerini datetime nesnesine dönüştürür."""
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            return isoparse(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _format_timestamp(ts):
    """Datetime nesnesini veya string'i ISO 8601 string formatına dönüştürür."""
    if isinstance(ts, str):
        return ts
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)


class IPBuffer:
    """
    src_ip bazlı 40-akışlık kayan pencere tamponu (IPBuffer).
    
    Özellikler:
    - Akışlar src_ip anahtarı altında gruplanır.
    - Aynı src_ip için iki akış arasındaki zaman farkı > timeout_s (300sn) ise tampon temizlenir (flush).
    - 40 akış dolduğunda pencere tamamlanmış sayılır (is_window_ready).
    - LSTM tahmini eşik değerinin (< threshold) altında kaldığında "Generic Attack" alarmı üretir.
    - Şemaya uygun alarm JSON çıktısı üretir: {src_ip, class, confidence, window_start, window_end}
    """

    def __init__(self, window_size: int = 40, timeout_s: float = 300.0, threshold: float = 0.50, class_map: dict = None):
        self.window_size = window_size
        self.timeout_s = timeout_s
        self.threshold = threshold
        self.class_map = class_map or {0: "Volumetric", 1: "WebAttack", 2: "Bot"}
        
        # src_ip -> deque of dicts: {"features": np.ndarray, "meta": dict, "ts": datetime}
        self.buffers = {}
        self.last_seen = {}

    def add_flow(self, src_ip: str, feature_vector: np.ndarray, meta: dict = None, timestamp=None):
        """
        Tampona yeni bir akış ekler.
        Zaman farkı > timeout_s (300sn) ise o src_ip için tamponu sıfırlar (flush).
        """
        ts_dt = _parse_timestamp(timestamp)
        meta = meta or {}
        meta["ts"] = ts_dt

        # 300sn GAP KONTROLÜ
        if src_ip in self.last_seen:
            prev_ts = self.last_seen[src_ip]
            time_gap = (ts_dt - prev_ts).total_seconds()
            if time_gap > self.timeout_s:
                # 300sn üzeri gap tespit edildi -> flush (tamponu temizle)
                self.flush_ip(src_ip)

        if src_ip not in self.buffers:
            self.buffers[src_ip] = deque(maxlen=self.window_size)

        self.buffers[src_ip].append({
            "features": feature_vector,
            "meta": meta,
            "ts": ts_dt
        })
        self.last_seen[src_ip] = ts_dt

    # Geriye dönük uyumluluk takma adı
    add = add_flow

    def flush_ip(self, src_ip: str):
        """Belirtilen src_ip için tamponu ve son görülme zamanını sıfırlar."""
        if src_ip in self.buffers:
            del self.buffers[src_ip]
        if src_ip in self.last_seen:
            del self.last_seen[src_ip]

    def flush_all(self):
        """Tüm tamponları temizler."""
        self.buffers.clear()
        self.last_seen.clear()

    def cleanup_stale(self, current_time=None):
        """300 saniyedir aktif olmayan tüm src_ip kayıtlarını temizler."""
        now = _parse_timestamp(current_time)
        stale_ips = [
            ip for ip, ts in self.last_seen.items()
            if (now - ts).total_seconds() > self.timeout_s
        ]
        for ip in stale_ips:
            self.flush_ip(ip)
        return len(stale_ips)

    cleanup = cleanup_stale

    def is_window_ready(self, src_ip: str) -> bool:
        """src_ip için 40 akış doldu mu kontrol eder."""
        return src_ip in self.buffers and len(self.buffers[src_ip]) == self.window_size

    def get_window_data(self, src_ip: str):
        """
        Tamamlanmış pencere verilerini döner.
        Return: (features_matrix, meta_list, window_start_str, window_end_str)
        """
        if not self.is_window_ready(src_ip):
            return None, None, None, None

        buf = list(self.buffers[src_ip])
        features_list = [item["features"] for item in buf]
        meta_list = [item["meta"] for item in buf]

        features_matrix = np.array(features_list, dtype=np.float32)
        window_start = _format_timestamp(buf[0]["ts"])
        window_end = _format_timestamp(buf[-1]["ts"])

        return features_matrix, meta_list, window_start, window_end

    def get_window(self, src_ip: str):
        """
        Geriye dönük uyumlu pencere alım metodu.
        70 core/enriched özniteliğe 8 zamansal behavioral özniteliği ekler (toplam 78 öznitelik).
        Return: (enhanced_features_matrix_78, (window_start, window_end)) veya (None, None)
        """
        if not self.is_window_ready(src_ip):
            return None, None

        features_matrix, meta_list, win_start, win_end = self.get_window_data(src_ip)
        win_features = compute_ip_window_features(meta_list)
        features_enhanced = np.column_stack([features_matrix] +
            [np.full(len(features_matrix), win_features[i]) for i in range(len(win_features))])
        return features_enhanced.astype(np.float32), (win_start, win_end)

    def generate_alarm(self, src_ip: str, probabilities: np.ndarray, window_start: str = None, window_end: str = None, clear_after: bool = True) -> dict:
        """
        LSTM tahmin olasılıkları (probabilities) ve eşik (threshold) mantığına göre alarm JSON nesnesi üretir.
        
        Threshold mantığı:
        - max(probabilities) >= self.threshold -> class_map'ten etiket alınır.
        - max(probabilities) < self.threshold  -> "Generic Attack" etiketi verilir (sınıflandırılır).
        
        Çıktı Şeması:
        {
            "src_ip": str,
            "class": str,
            "confidence": float,
            "window_start": str,
            "window_end": str
        }
        """
        probs = np.asarray(probabilities, dtype=np.float32).flatten()
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        if confidence < self.threshold:
            predicted_class = "Generic Attack"
        else:
            predicted_class = self.class_map.get(max_idx, "Generic Attack")

        if window_start is None or window_end is None:
            _, _, ws, we = self.get_window_data(src_ip)
            window_start = window_start or ws or ""
            window_end = window_end or we or ""

        alarm = {
            "src_ip": src_ip,
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "window_start": window_start,
            "window_end": window_end
        }

        if clear_after:
            self.flush_ip(src_ip)

        return alarm

    def process_and_generate_alarm(self, src_ip: str, predict_fn, clear_after: bool = True) -> dict:
        """
        LSTM tahmin fonksiyonunu (predict_fn) pencereye uygulayıp alarm nesnesi üretir.
        predict_fn: callable, (features_matrix, meta_list) -> probabilities array
        """
        features_matrix, meta_list, window_start, window_end = self.get_window_data(src_ip)
        if features_matrix is None:
            return None

        probs = predict_fn(features_matrix, meta_list)
        return self.generate_alarm(
            src_ip=src_ip,
            probabilities=probs,
            window_start=window_start,
            window_end=window_end,
            clear_after=clear_after
        )
