#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ip_buffer.py - IPBuffer Modülü Unit Testleri
=================================================
Test Kapsamı:
1. Pencere Boyutu (40 Akış) ve Kayan Pencere Davranışı
2. src_ip Bazlı Bağımsız Gruplama
3. 300 Saniye Zaman Farkı (Gap) ile Tampon Temizleme (Flush)
4. LSTM Çıkışı Eşik (Threshold) Mantığı (Eşik Altı -> "Generic Attack")
5. Alarm JSON Şeması Doğrulaması {src_ip, class, confidence, window_start, window_end}
"""

import unittest
import json
from datetime import datetime, timedelta, timezone
import numpy as np

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from ip_buffer import IPBuffer


class TestIPBuffer(unittest.TestCase):

    def setUp(self):
        """Her test öncesi 40 akışlık, 300sn timeout'lu IPBuffer örneği oluşturur."""
        self.buffer = IPBuffer(window_size=40, timeout_s=300.0, threshold=0.50)
        self.dummy_features = np.ones(70, dtype=np.float32)

    def test_window_accumulation_40_flows(self):
        """Sentetik 40 akış ekleme ve window_ready durumunu test eder."""
        src_ip = "192.168.1.100"
        start_time = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)

        # İlk 39 akış ekleniyor
        for i in range(39):
            ts = start_time + timedelta(seconds=i * 2)  # 2sn aralıklarla
            self.buffer.add_flow(src_ip, self.dummy_features, meta={"flow_id": i}, timestamp=ts)
            self.assertFalse(
                self.buffer.is_window_ready(src_ip),
                f"Pencere {i+1}. akışta dolmamış olmalı."
            )

        # 40. akış ekleniyor
        ts_40 = start_time + timedelta(seconds=39 * 2)
        self.buffer.add_flow(src_ip, self.dummy_features, meta={"flow_id": 39}, timestamp=ts_40)

        # 40 akış doldu, ready olmalı
        self.assertTrue(
            self.buffer.is_window_ready(src_ip),
            "40 akış tamamlandığında pencere hazır (ready) olmalıdır."
        )

        features_matrix, meta_list, win_start, win_end = self.buffer.get_window_data(src_ip)
        self.assertEqual(features_matrix.shape, (40, 70))
        self.assertEqual(len(meta_list), 40)
        self.assertEqual(win_start, start_time.isoformat())
        self.assertEqual(win_end, ts_40.isoformat())

    def test_grouping_by_src_ip(self):
        """Akışların src_ip bazlı tamamen bağımsız gruplandığını test eder."""
        ip_a = "10.0.0.1"
        ip_b = "10.0.0.2"
        base_time = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)

        # IP A'ya 40 akış ekle
        for i in range(40):
            ts = base_time + timedelta(seconds=i)
            self.buffer.add_flow(ip_a, self.dummy_features, meta={"ip": ip_a}, timestamp=ts)

        # IP B'ye 15 akış ekle
        for i in range(15):
            ts = base_time + timedelta(seconds=i)
            self.buffer.add_flow(ip_b, self.dummy_features, meta={"ip": ip_b}, timestamp=ts)

        # IP A hazır olmalı, IP B hazır olmamalı
        self.assertTrue(self.buffer.is_window_ready(ip_a))
        self.assertFalse(self.buffer.is_window_ready(ip_b))
        self.assertEqual(len(self.buffer.buffers[ip_b]), 15)

    def test_flush_on_300s_gap(self):
        """İki akış arasında >300 saniye zaman farkı oluştuğunda tamponun sıfırlandığını (flush) test eder."""
        src_ip = "172.16.0.5"
        base_time = datetime(2026, 7, 24, 14, 0, 0, tzinfo=timezone.utc)

        # 30 akış ekle (her biri 1sn arayla)
        for i in range(30):
            ts = base_time + timedelta(seconds=i)
            self.buffer.add_flow(src_ip, self.dummy_features, timestamp=ts)

        self.assertEqual(len(self.buffer.buffers[src_ip]), 30)

        # 305 saniye sonra yeni bir akış ekle (>300sn gap)
        ts_after_gap = base_time + timedelta(seconds=30 + 305)
        self.buffer.add_flow(src_ip, self.dummy_features, timestamp=ts_after_gap)

        # 300sn üzeri gap nedeniyle eski 30 akış silinmeli, sadece yeni akış olmalı
        self.assertEqual(
            len(self.buffer.buffers[src_ip]), 1,
            ">300sn gap sonrası tampon sıfırlanmalı ve sadece yeni akış bulunmalıdır."
        )
        self.assertFalse(self.buffer.is_window_ready(src_ip))

    def test_cleanup_stale(self):
        """Inaktif kalmış IP'lerin cleanup_stale ile temizlenmesini test eder."""
        ip_stale = "192.168.1.50"
        t0 = datetime(2026, 7, 24, 8, 0, 0, tzinfo=timezone.utc)
        self.buffer.add_flow(ip_stale, self.dummy_features, timestamp=t0)

        # 400 saniye sonra cleanup çalıştır
        t_now = t0 + timedelta(seconds=400)
        removed_count = self.buffer.cleanup_stale(current_time=t_now)

        self.assertEqual(removed_count, 1)
        self.assertNotIn(ip_stale, self.buffer.buffers)

    def test_lstm_threshold_above_threshold(self):
        """Eşik değerinin üzerindeki LSTM tahmininde ilgili sınıf etiketinin verildiğini test eder."""
        src_ip = "192.168.1.200"
        t0 = datetime(2026, 7, 24, 15, 0, 0, tzinfo=timezone.utc)

        for i in range(40):
            self.buffer.add_flow(src_ip, self.dummy_features, timestamp=t0 + timedelta(seconds=i))

        # Olasılıklar: [0.85, 0.10, 0.05] (Sınıf 0: Volumetric, Güven: %85 > %50)
        probs = np.array([0.85, 0.10, 0.05], dtype=np.float32)
        alarm = self.buffer.generate_alarm(src_ip, probs)

        self.assertEqual(alarm["src_ip"], src_ip)
        self.assertEqual(alarm["class"], "Volumetric")
        self.assertEqual(alarm["confidence"], 0.85)
        self.assertEqual(alarm["window_start"], t0.isoformat())
        self.assertEqual(alarm["window_end"], (t0 + timedelta(seconds=39)).isoformat())

    def test_lstm_threshold_below_threshold_generic_attack(self):
        """
        CRITICAL TEST: Eşik altındaki LSTM tahmininde 'Generic Attack' etiketi atandığını test eder.
        Bu durum bir discard/fallback değil, alarm JSON'a 'Generic Attack' sınıfı olarak yazılır.
        """
        src_ip = "192.168.1.201"
        t0 = datetime(2026, 7, 24, 16, 0, 0, tzinfo=timezone.utc)

        for i in range(40):
            self.buffer.add_flow(src_ip, self.dummy_features, timestamp=t0 + timedelta(seconds=i))

        # Olasılıklar: [0.42, 0.38, 0.20] (En yüksek olasılık %42 < %50 eşik)
        probs = np.array([0.42, 0.38, 0.20], dtype=np.float32)
        alarm = self.buffer.generate_alarm(src_ip, probs)

        self.assertEqual(alarm["src_ip"], src_ip)
        self.assertEqual(
            alarm["class"], "Generic Attack",
            "Eşik altındaki tahminde etiket 'Generic Attack' olmalıdır."
        )
        self.assertEqual(alarm["confidence"], 0.42)
        self.assertIn("window_start", alarm)
        self.assertIn("window_end", alarm)

    def test_alarm_json_schema(self):
        """Alarm nesnesinin tam istenen JSON şemasıyla eşleştiğini doğrular."""
        src_ip = "10.10.10.10"
        t0 = datetime(2026, 7, 24, 18, 0, 0, tzinfo=timezone.utc)

        for i in range(40):
            self.buffer.add_flow(src_ip, self.dummy_features, timestamp=t0 + timedelta(seconds=i))

        probs = np.array([0.15, 0.80, 0.05], dtype=np.float32)  # WebAttack (%80)
        alarm = self.buffer.generate_alarm(src_ip, probs)

        # Gerekli anahtarlar: {src_ip, class, confidence, window_start, window_end}
        expected_keys = {"src_ip", "class", "confidence", "window_start", "window_end"}
        self.assertEqual(set(alarm.keys()), expected_keys)

        # JSON serileştirilebilir mi?
        json_output = json.dumps(alarm)
        parsed = json.loads(json_output)

        self.assertEqual(parsed["src_ip"], "10.10.10.10")
        self.assertEqual(parsed["class"], "WebAttack")
        self.assertEqual(parsed["confidence"], 0.80)
        self.assertIsInstance(parsed["window_start"], str)
        self.assertIsInstance(parsed["window_end"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
