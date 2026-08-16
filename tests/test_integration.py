#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_integration.py - Uçtan Uca Entegrasyon Testi (End-to-End Pipeline)
======================================================================
Pipeline Akışı:
  extract_features_v7 (70 öznitelik)
       │
       ▼
  XGBoost Sınıflandırma (70 öznitelik)
       │ (Saldırı + DoS/DDoS Olmayan)
       ▼
  IPBuffer (40 akış biriktirme + 8 davranışsal öznitelik → 78 öznitelik)
       │
       ▼
  LSTM Çıkarımı (1, 40, 78)
       │
       ▼
  Eşik Kontrolü & Alarm JSON Çıktısı {src_ip, class, confidence, window_start, window_end}

Ara Katman Şekil ve Tip Doğrulamaları:
1. extract_features_v7: shape (1, 70), float32
2. XGBoost Scaler & Model Input: 70 öznitelik
3. IPBuffer Window Data: shape (40, 70)
4. Window Behavioral Features: shape (8,)
5. LSTM Input Matrix: shape (40, 78)
6. LSTM Reshaped 3D Tensor: shape (1, 40, 78), float16/float32
7. LSTM Output Probs: shape (1, 3)
8. Alarm JSON Şeması: {src_ip, class, confidence, window_start, window_end}
"""

import unittest
import json
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np

# 'pipeline' dizinini import yoluna ekle
PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import tensorflow as tf
try:
    # GPU/BLAS stream çakışmalarını önlemek için TensorFlow CPU moduna ayarla
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
from hybrid_inference import FlowEnrichment, extract_features_v7
from features import compute_ip_window_features
from ip_buffer import IPBuffer, _parse_timestamp


class TestEndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Gerçek model dosyalarını ve eve.json verisini yükler."""
        cls.root_dir = Path(__file__).parent.parent
        cls.eve_path = cls.root_dir / "data" / "eve" / "eve.json"
        if not cls.eve_path.exists():
            cls.eve_path = cls.root_dir / "data" / "eve" / "eve_test20.json"
        cls.xgb_model_path = cls.root_dir / "models" / "ids_model_v8_final.pkl"
        if not cls.xgb_model_path.exists():
            cls.xgb_model_path = cls.root_dir / "archive" / "models" / "ids_model_v7_final.pkl"
        cls.lstm_model_path = cls.root_dir / "pipeline" / "lstm_best.keras"
        cls.lstm_scaler_path = cls.root_dir / "pipeline" / "lstm_scaler.pkl"

        # Dosya mevcudiyet kontrolleri
        cls.assertTrue(cls.eve_path.exists(), f"{cls.eve_path} bulunamadı.")
        cls.assertTrue(cls.xgb_model_path.exists(), f"{cls.xgb_model_path} bulunamadı.")
        cls.assertTrue(cls.lstm_model_path.exists(), f"{cls.lstm_model_path} bulunamadı.")
        cls.assertTrue(cls.lstm_scaler_path.exists(), f"{cls.lstm_scaler_path} bulunamadı.")

        # XGBoost Yükleme
        with open(cls.xgb_model_path, "rb") as f:
            xgb_b = pickle.load(f)
        cls.xgb_model = xgb_b["model"]
        cls.xgb_scaler = xgb_b["scaler"]
        cls.xgb_threshold = xgb_b.get("threshold", 0.59)
        cls.xgb_inv_class_map = xgb_b.get("inv_class_map", {})

        # CPU Uyum Ayarları
        try:
            cls.xgb_model.set_params(device="cpu")
            cls.xgb_model.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass

        # LSTM Yükleme
        cls.lstm_model = tf.keras.models.load_model(cls.lstm_model_path, compile=False)
        with open(cls.lstm_scaler_path, "rb") as f:
            cls.lstm_scaler = pickle.load(f)

    def test_layer1_feature_extraction_shape_and_type(self):
        """1. Katman: extract_features_v7 çıktısının (1, 70) float32 tipinde olduğunu doğrular."""
        flow_cache = FlowEnrichment()
        sample_event = None

        with open(self.eve_path, "r", encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line)
                etype = ev.get("event_type")
                if etype in ("http", "tls", "dns"):
                    flow_cache.ingest(ev)
                elif etype == "flow":
                    sample_event = ev
                    break

        self.assertIsNotNone(sample_event, "eve.json içinde flow event bulunamadı.")
        enr = flow_cache.get(sample_event.get("flow_id"))
        features, meta = extract_features_v7(sample_event, enr)

        self.assertIsNotNone(features, "Öznitelik çıkarımı None döndü.")
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (1, 70), f"Beklenen şekil (1, 70), alınan: {features.shape}")
        self.assertEqual(features.dtype, np.float32)
        self.assertIn("src_ip", meta)
        self.assertIn("dest_ip", meta)

    @staticmethod
    def _predict_xgb_prob(model, scaled_feat):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(scaled_feat)[0]
        import xgboost as xgb
        return model.predict(xgb.DMatrix(scaled_feat))[0]

    def test_layer2_xgboost_scaler_and_prediction(self):
        """2. Katman: XGBoost scaler ve model tahmin şekillerinin (1, 70) ve 6 sınıf olasılık olduğunu doğrular."""
        dummy_feat = np.ones((1, 70), dtype=np.float32)
        scaled_feat = self.xgb_scaler.transform(dummy_feat)
        self.assertEqual(scaled_feat.shape, (1, 70), f"XGBoost scaler (1, 70) dönmeli. Alınan: {scaled_feat.shape}")

        probs = self._predict_xgb_prob(self.xgb_model, scaled_feat)
        self.assertEqual(len(probs), 6, f"XGBoost 6 sınıf olasılığı dönmelidir. Alınan sınıf sayısı: {len(probs)}")
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_layer3_ipbuffer_and_behavioral_features(self):
        """3. Katman: IPBuffer 40 akış topladıktan sonra 70 base + 8 behavioral = 78 öznitelik matriksi ürettiğini doğrular."""
        ip_buffer = IPBuffer(window_size=40, timeout_s=300.0)
        src_ip = "192.168.1.55"
        base_time = datetime(2026, 7, 24, 10, 0, 0)

        # 40 sentetik akış ekle
        for i in range(40):
            feat_70 = np.full(70, float(i), dtype=np.float32)
            meta = {
                "ts": base_time + timedelta(seconds=i * 2),
                "dest_ip": "10.0.0.1",
                "dest_port": 80,
                "total_bytes": 500 + i * 10,
                "dns_queries": ["example.com"],
                "http_uri": f"/api/v1/test_{i}",
                "tls_sni": "example.com"
            }
            ip_buffer.add_flow(src_ip, feat_70, meta, timestamp=meta["ts"])

        self.assertTrue(ip_buffer.is_window_ready(src_ip))

        features_matrix, meta_list, win_start, win_end = ip_buffer.get_window_data(src_ip)
        self.assertEqual(features_matrix.shape, (40, 70))
        self.assertEqual(len(meta_list), 40)

        # 8 Davranışsal öznitelik hesapla
        win_features = compute_ip_window_features(meta_list)
        self.assertEqual(win_features.shape, (8,), f"Davranışsal öznitelik vektörü (8,) olmalıdır. Alınan: {win_features.shape}")

        # 70 + 8 = 78 öznitelik matriks birleştirme
        features_78, (ws, we) = ip_buffer.get_window(src_ip)
        self.assertEqual(features_78.shape, (40, 78), f"LSTM öncesi öznitelik matriksi (40, 78) olmalıdır. Alınan: {features_78.shape}")
        self.assertEqual(features_78.dtype, np.float32)

    def test_layer4_lstm_input_shape_t40_compatibility(self):
        """4. Katman: LSTM modelinin (1, 40, 78) girdi tensor boyutunu ve (1, 3) çıktı olasılıklarını doğrular."""
        features_78 = np.random.randn(40, 78).astype(np.float32)

        # Scaler dönüşümü (40, 78) -> (40, 78)
        scaled_78 = self.lstm_scaler.transform(features_78)
        self.assertEqual(scaled_78.shape, (40, 78))

        # 3D Tensor hazırlama: (1, 40, 78)
        input_3d = scaled_78.reshape(1, 40, 78).astype(np.float32)
        self.assertEqual(input_3d.shape, (1, 40, 78), f"LSTM girdi tensorü (1, 40, 78) olmalıdır. Alınan: {input_3d.shape}")

        # LSTM Tahmini
        lstm_probs = self.lstm_model.predict(input_3d, verbose=0)[0]
        self.assertEqual(len(lstm_probs), 3, f"LSTM 3 sınıf olasılığı dönmelidir. Alınan: {len(lstm_probs)}")
        self.assertAlmostEqual(float(np.sum(lstm_probs)), 1.0, places=3)

    def test_end_to_end_pipeline_with_real_eve_json(self):
        """
        Uçtan Uca Bütünsel Entegrasyon Testi:
        Real eve.json -> extract_features_v7 -> XGBoost -> IPBuffer -> LSTM -> Alarm JSON
        """
        ip_buffer = IPBuffer(window_size=40, timeout_s=300.0, threshold=0.50)
        flow_cache = FlowEnrichment()
        generated_alarms = []

        flow_count = 0
        with open(self.eve_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev = json.loads(line)
                etype = ev.get("event_type")

                if etype in ("http", "tls", "dns"):
                    flow_cache.ingest(ev)
                    continue

                if etype != "flow":
                    continue

                flow_count += 1
                enr = flow_cache.get(ev.get("flow_id"))
                features, meta = extract_features_v7(ev, enr)

                if features is None:
                    continue

                # 1) XGBoost Aşaması
                scaled_feat = self.xgb_scaler.transform(features)
                xgb_probs = self._predict_xgb_prob(self.xgb_model, scaled_feat)
                prob_atk = 1.0 - xgb_probs[0]
                xgb_pred = int(xgb_probs.argmax())

                # Benign ise geç
                if xgb_pred == 0 or prob_atk < self.xgb_threshold:
                    continue

                src_ip = meta["src_ip"]

                # 2) DoS / DDoS Bypass (XGBoost Yeterli)
                if xgb_pred in (1, 2):
                    alarm = {
                        "src_ip": src_ip,
                        "class": "DoS/DDoS",
                        "confidence": round(float(prob_atk), 4),
                        "window_start": meta["timestamp"],
                        "window_end": meta["timestamp"]
                    }
                    generated_alarms.append(alarm)
                    continue

                # 3) Diğer Saldırılar -> IPBuffer'a ekle
                ip_meta = {
                    "ts": _parse_timestamp(meta["timestamp"]),
                    "dest_ip": meta["dest_ip"],
                    "dest_port": meta["dest_port"],
                    "total_bytes": meta["total_bytes"],
                    "dns_queries": [],
                    "http_uri": None,
                    "tls_sni": None
                }
                ip_buffer.add_flow(src_ip, features[0], ip_meta, timestamp=ip_meta["ts"])

                # 4) 40 Akış Dolduysa LSTM'e Gönder
                if ip_buffer.is_window_ready(src_ip):
                    win_features_78, (win_start, win_end) = ip_buffer.get_window(src_ip)
                    self.assertIsNotNone(win_features_78)
                    self.assertEqual(win_features_78.shape, (40, 78))

                    # LSTM Normalizasyonu ve Reshape (1, 40, 78)
                    win_scaled = self.lstm_scaler.transform(win_features_78)
                    win_3d = win_scaled.reshape(1, 40, 78).astype(np.float32)

                    # LSTM Tahmini
                    lstm_probs = self.lstm_model.predict(win_3d, verbose=0)[0]
                    alarm = ip_buffer.generate_alarm(
                        src_ip=src_ip,
                        probabilities=lstm_probs,
                        window_start=win_start,
                        window_end=win_end,
                        clear_after=True
                    )
                    generated_alarms.append(alarm)

        print(f"\n[INTEGRATION SUCCESS] {flow_count} flow işlendi, {len(generated_alarms)} alarm üretildi.")

        # Üretilen her alarmın JSON şeması doğrulama
        for alarm in generated_alarms:
            self.assertIn("src_ip", alarm)
            self.assertIn("class", alarm)
            self.assertIn("confidence", alarm)
            self.assertIn("window_start", alarm)
            self.assertIn("window_end", alarm)
            self.assertIn(alarm["class"], ["Volumetric", "WebAttack", "Bot", "Generic Attack", "DoS/DDoS"])
            # JSON serileştirilebilir mi?
            json_str = json.dumps(alarm)
            self.assertTrue(len(json_str) > 0)

    def test_coral_vs_non_coral_comparison(self):
        """
        CORAL Karşılaştırma Testi:
        Aynı 17.420 akışlık eve.json verisetini CORAL'sız (Before) ve CORAL'lı (After)
        iki ayrı çalıştırarak alarm sayılarını ve sınıf dağılımlarını karşılaştırır.
        """
        from collections import Counter
        from coral_domain_adaptation import CORALDomainAdapter
        from hybrid_inference import hybrid_predict

        coral_adapter_path = self.root_dir / "pipeline" / "coral_adapter.pkl"
        self.assertTrue(coral_adapter_path.exists(), "coral_adapter.pkl bulunamadı.")
        coral_adapter = CORALDomainAdapter.load(str(coral_adapter_path))

        def run_pipeline(use_coral=False):
            ip_buf = IPBuffer(window_size=40, timeout_s=300.0, threshold=0.50)
            cache = FlowEnrichment()
            alarms = []
            adapter = coral_adapter if use_coral else None

            with open(self.eve_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    etype = ev.get("event_type")

                    if etype in ("http", "tls", "dns"):
                        cache.ingest(ev)
                        continue
                    if etype != "flow":
                        continue

                    enr = cache.get(ev.get("flow_id"))
                    alert = hybrid_predict(
                        ev, enr,
                        self.xgb_model, self.xgb_scaler, self.xgb_threshold,
                        self.xgb_inv_class_map, {}, True,
                        self.lstm_model, self.lstm_scaler, {},
                        ip_buf, coral_adapter=adapter
                    )
                    if alert:
                        alarms.append(alert["ai"]["label"])
            return Counter(alarms)

        print("\n[RUN 1/2] CORAL'sız Akış Çalıştırılıyor...")
        counts_before = run_pipeline(use_coral=False)

        print("[RUN 2/2] CORAL Domain Adaptation Alignment ile Akış Çalıştırılıyor...")
        counts_after = run_pipeline(use_coral=True)

        all_classes = sorted(list(set(counts_before.keys()) | set(counts_after.keys())))
        total_before = sum(counts_before.values())
        total_after = sum(counts_after.values())

        print("\n" + "=" * 78)
        print("CORAL DOMAIN ADAPTATION COMPARISON TABLE (17,420 flows - eve.json)")
        print("=" * 78)
        print(f"{'Metric / Attack Class':<30} {'Before CORAL':>15} {'After CORAL':>15} {'Difference':>14}")
        print("-" * 78)
        print(f"{'TOTAL ALARMS GENERATED':<30} {total_before:>15d} {total_after:>15d} {total_after - total_before:>+14d}")
        print("-" * 78)
        for cls in all_classes:
            b_cnt = counts_before.get(cls, 0)
            a_cnt = counts_after.get(cls, 0)
            diff = a_cnt - b_cnt
            print(f"{cls:<30} {b_cnt:>15d} {a_cnt:>15d} {diff:>+14d}")
        print("=" * 78)

        # Doğrulama: Çalıştırma çıktıları geçerli
        self.assertGreater(total_before, 0)
        self.assertGreater(total_after, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
