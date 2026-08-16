#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORAL Domain Adaptation Evaluation — Before/After XGBoost FP Rate Comparison
=============================================================================
Source: CICIDS2018 full_dataset (training data distribution)
Target: test_20 (test data with labels)
"""

import json
import pickle
import numpy as np
from datetime import datetime

import feature_extract_standalone as fe


def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle.get('threshold', 0.66), bundle.get('inv_class_map', {})


def evaluate_fp_rate(model, scaler, threshold, X_benign):
    """Evaluate false positive rate on benign traffic."""
    X_scaled = scaler.transform(X_benign)
    probs = model.predict_proba(X_scaled)
    prob_atk = 1.0 - probs[:, 0]  # Class 0 = Benign
    preds = (prob_atk >= threshold).astype(int)
    fp_rate = preds.mean()
    return fp_rate, preds


def run_coral_comparison():
    print("=" * 70)
    print("CORAL DOMAIN ADAPTATION — BEFORE/AFTER EVALUATION")
    print("=" * 70)
    
    # Paths
    model_path = "/run/media/mehmet/siber data1/ai modeli xgboost/ids_model_v7_final.pkl"
    source_eve = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/full_dataset/eve.json"
    target_eve = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/test_20/eve_test20.json"
    target_labels = "/run/media/mehmet/siber data1/ai modeli xgboost/eve_labeled/test_20/labels.json"
    
    # Load model
    print(f"\n[1/6] Loading XGBoost model...")
    model, scaler, threshold, inv_class_map = load_model(model_path)
    print(f"    Threshold: {threshold}")
    print(f"    Classes: {inv_class_map}")
    
    # Load target labels
    print(f"\n[2/6] Loading target labels...")
    with open(target_labels) as f:
        label_data = json.load(f)
    label_fn = fe.create_label_fn(label_data['attack_windows'], label_data['victim_ips'])
    
    # Extract source features (training distribution)
    print(f"\n[3/6] Extracting source features (max 50k)...")
    X_source, _ = fe.load_eve_features(source_eve, max_samples=50000)
    print(f"    Source shape: {X_source.shape}")
    
    # Extract target features with labels
    print(f"\n[4/6] Extracting target features with labels...")
    X_target, y_target = fe.load_eve_features(target_eve, label_fn=label_fn)
    print(f"    Target shape: {X_target.shape}")
    print(f"    Benign: {(y_target==0).sum()}, Attack: {(y_target==1).sum()}")
    
    # Split benign for FP evaluation
    X_target_benign = X_target[y_target == 0]
    print(f"    Benign target samples: {len(X_target_benign)}")
    
    # BEFORE CORAL
    print(f"\n[5/6] Evaluating XGBoost FP rate BEFORE CORAL...")
    fp_before, preds_before = evaluate_fp_rate(model, scaler, threshold, X_target_benign)
    print(f"    FP Rate (before): {fp_before:.4f} ({fp_before*100:.2f}%)")
    print(f"    FP Count: {int(fp_before*len(X_target_benign))} / {len(X_target_benign)}")
    
    # CORAL Adaptation
    print(f"\n[6/6] Fitting CORAL adapter...")
    from coral_domain_adaptation import CORALDomainAdapter
    adapter = CORALDomainAdapter(lambda_reg=1e-5)
    adapter.fit(X_source, X_target, scale=True)
    metrics = adapter.get_metrics()
    print(f"    Frobenius distance: {metrics['frobenius_distance']:.6f}")
    print(f"    Trace ratio (Ct/Cs): {metrics['trace_ratio_target_over_source']:.6f}")
    print(f"    Cond. number (source): {metrics['condition_number_source']:.2f}")
    print(f"    Cond. number (target): {metrics['condition_number_target']:.2f}")
    
    # Transform target benign to source space
    X_target_benign_aligned = adapter.transform_target(X_target_benign)
    
    # AFTER CORAL
    print(f"\n[7/7] Evaluating XGBoost FP rate AFTER CORAL...")
    fp_after, preds_after = evaluate_fp_rate(model, scaler, threshold, X_target_benign_aligned)
    print(f"    FP Rate (after): {fp_after:.4f} ({fp_after*100:.2f}%)")
    print(f"    FP Count: {int(fp_after*len(X_target_benign))} / {len(X_target_benign)}")
    
    # Results table
    improvement = (fp_before - fp_after) / fp_before * 100 if fp_before > 0 else 0
    fp_count_before = int(fp_before * len(X_target_benign))
    fp_count_after = int(fp_after * len(X_target_benign))
    
    print("\n" + "=" * 70)
    print("RESULTS: BEFORE vs AFTER CORAL ALIGNMENT")
    print("=" * 70)
    print(f"{'Metric':<35} {'Before':>12} {'After':>12} {'Improvement':>12}")
    print("-" * 70)
    print(f"{'FP Rate (Benign target)':<35} {fp_before:>11.4f} {fp_after:>11.4f} {improvement:>11.1f}%")
    print(f"{'FP Count':<35} {fp_count_before:>12d} {fp_count_after:>12d} {fp_count_before - fp_count_after:>11d}")
    print(f"{'Frobenius ||Cs-Ct||_F':<35} {'N/A':>12} {metrics['frobenius_distance']:>12.6f} {'-':>12}")
    print(f"{'Trace Ratio (Ct/Cs)':<35} {'N/A':>12} {metrics['trace_ratio_target_over_source']:>12.6f} {'-':>12}")
    print(f"{'Cond. Number (Source)':<35} {'N/A':>12} {metrics['condition_number_source']:>12.2f} {'-':>12}")
    print(f"{'Cond. Number (Target)':<35} {'N/A':>12} {metrics['condition_number_target']:>12.2f} {'-':>12}")
    print("=" * 70)
    
    # Save results
    results = {
        'fp_before': float(fp_before),
        'fp_after': float(fp_after),
        'improvement_pct': float(improvement),
        'fp_count_before': fp_count_before,
        'fp_count_after': fp_count_after,
        'frobenius': float(metrics['frobenius_distance']),
        'trace_ratio': float(metrics['trace_ratio_target_over_source']),
        'cond_source': float(metrics['condition_number_source']),
        'cond_target': float(metrics['condition_number_target']),
        'n_benign': int(len(X_target_benign)),
        'n_source': int(len(X_source)),
        'threshold': float(threshold),
    }
    
    with open('/run/media/mehmet/siber data1/ai modeli xgboost/coral_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to coral_eval_results.json")
    
    return results


if __name__ == '__main__':
    run_coral_comparison()