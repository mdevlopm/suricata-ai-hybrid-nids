#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORAL Domain Adaptation Evaluation — Before/After XGBoost FP Rate Comparison
=============================================================================
Source: CICIDS2018 full_dataset (training data distribution)
Target: test_20 / test_190 (test data with labels)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import importlib.util

# Load features module - extract_features_v7 is in hybrid_inference.py
spec = importlib.util.spec_from_file_location("hybrid_inference", 
    "/run/media/mehmet/siber data1/ai modeli xgboost/model eğitim dosyaları/hybrid_inference.py")
hybrid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid_module)
extract_features_v7 = hybrid_module.extract_features_v7

# Load CORAL adapter
spec2 = importlib.util.spec_from_file_location("coral", 
    "/run/media/mehmet/siber data1/ai modeli xgboost/coral_domain_adaptation.py")
coral_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(coral_module)
CORALDomainAdapter = coral_module.CORALDomainAdapter


def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle.get('threshold', 0.66), bundle.get('inv_class_map', {})


def create_label_fn(attack_windows, victim_ips):
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


def extract_labeled_features(eve_path, label_fn, max_samples=None):
    X_list, y_list = [], []
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
                    feat = extract_features_v7(event)
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(label_fn(event))
                        count += 1
            except json.JSONDecodeError:
                continue
    return np.stack(X_list).astype(np.float32), np.array(y_list)


def evaluate_fp_rate(model, scaler, threshold, X_benign):
    """False positive rate on benign samples."""
    X_scaled = scaler.transform(X_benign)
    probs = model.predict_proba(X_scaled)
    prob_atk = 1.0 - probs[:, 0]
    preds = (prob_atk >= threshold).astype(int)
    return preds.mean(), preds


def main():
    print("=" * 75)
    print("CORAL DOMAIN ADAPTATION — BEFORE/AFTER FP RATE EVALUATION")
    print("=" * 75)
    
    # Paths
    model_path = "/run/media/mehmet/siber data1/ai modeli xgboost/ids_model_v7_final.pkl"
    source_eve = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/full_dataset/eve.json"
    target_eve = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/test_20/eve_test20.json"
    target_labels = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/test_20/labels.json"
    
    # Load model
    print("\n[1/6] Loading XGBoost v7 model...")
    model, scaler, threshold, inv_class_map = load_model(model_path)
    print(f"    Threshold: {threshold}")
    print(f"    Classes: {inv_class_map}")
    
    # Load labels
    print("\n[2/6] Loading target labels...")
    with open(target_labels) as f:
        label_data = json.load(f)
    label_fn = create_label_fn(label_data['attack_windows'], label_data['victim_ips'])
    
    # Extract source features (subset for speed)
    print("\n[3/6] Extracting source features (50k from full_dataset)...")
    X_source, _ = extract_labeled_features(source_eve, lambda e: 0, max_samples=50000)
    print(f"    Source shape: {X_source.shape}")
    
    # Extract target features with labels
    print("\n[4/6] Extracting target features (test_20)...")
    X_target, y_target = extract_labeled_features(target_eve, label_fn)
    print(f"    Target shape: {X_target.shape}")
    print(f"    Benign: {(y_target==0).sum()}, Attack: {(y_target==1).sum()}")
    
    X_target_benign = X_target[y_target == 0]
    print(f"    Benign samples for FP eval: {len(X_target_benign)}")
    
    # BEFORE CORAL
    print("\n[5/6] Evaluating XGBoost FP rate BEFORE CORAL...")
    fp_before, preds_before = evaluate_fp_rate(model, scaler, threshold, X_target_benign)
    print(f"    FP Rate: {fp_before:.4f} ({fp_before*100:.2f}%)")
    print(f"    FP Count: {int(fp_before * len(X_target_benign))} / {len(X_target_benign)}")
    
    # CORAL Adaptation
    print("\n[6/6] Fitting CORAL adapter (source -> target)...")
    adapter = CORALDomainAdapter(lambda_reg=1e-5)
    adapter.fit(X_source, X_target, scale=True)
    metrics = adapter.get_metrics()
    print(f"    Frobenius ||Cs-Ct||_F: {metrics['frobenius_distance']:.6f}")
    print(f"    Trace ratio Ct/Cs: {metrics['trace_ratio_target_over_source']:.6f}")
    print(f"    Cond # Source: {metrics['condition_number_source']:.2f}")
    print(f"    Cond # Target: {metrics['condition_number_target']:.2f}")
    
    # Transform target benign to source space
    X_target_benign_aligned = adapter.transform_target(X_target_benign)
    
    # AFTER CORAL
    print("\nEvaluating XGBoost FP rate AFTER CORAL alignment...")
    fp_after, preds_after = evaluate_fp_rate(model, scaler, threshold, X_target_benign_aligned)
    print(f"    FP Rate: {fp_after:.4f} ({fp_after*100:.2f}%)")
    print(f"    FP Count: {int(fp_after * len(X_target_benign))} / {len(X_target_benign)}")
    
    # Results table
    improvement = (fp_before - fp_after) / fp_before * 100 if fp_before > 0 else 0
    
    print("\n" + "=" * 75)
    print("RESULTS: BEFORE vs AFTER CORAL")
    print("=" * 75)
    print(f"{'Metric':<40} {'Before':>12} {'After':>12} {'Improvement':>12}")
    print("-" * 75)
    print(f"{'FP Rate (Benign target)':<40} {fp_before:>11.4f} {fp_after:>11.4f} {improvement:>11.1f}%")
    print(f"{'FP Count':<40} {int(fp_before*len(X_target_benign)):>12d} {int(fp_after*len(X_target_benign)):>12d} {int(fp_before*len(X_target_benign)) - int(fp_after*len(X_target_benign)):>11d}")
    print(f"{'Frobenius ||Cs-Ct||_F':<40} {'N/A':>12} {metrics['frobenius_distance']:>12.6f} {'-':>12}")
    print(f"{'Trace Ratio (Ct/Cs)':<40} {'N/A':>12} {metrics['trace_ratio_target_over_source']:>12.6f} {'-':>12}")
    print(f"{'Condition Number (Source)':<40} {'N/A':>12} {metrics['condition_number_source']:>12.2f} {'-':>12}")
    print(f"{'Condition Number (Target)':<40} {'N/A':>12} {metrics['condition_number_target']:>12.2f} {'-':>12}")
    print("=" * 75)
    
    return {
        'fp_before': float(fp_before),
        'fp_after': float(fp_after),
        'improvement_pct': float(improvement),
        'frobenius': float(metrics['frobenius_distance']),
        'trace_ratio': float(metrics['trace_ratio_target_over_source']),
        'n_benign': int(len(X_target_benign)),
        'n_source': int(len(X_source)),
    }


if __name__ == '__main__':
    results = main()
    import json
    with open('/run/media/mehmet/siber data1/ai modeli xgboost/coral_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to coral_eval_results.json")