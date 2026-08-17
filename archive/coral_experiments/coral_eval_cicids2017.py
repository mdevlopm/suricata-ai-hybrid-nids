#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORAL Evaluation on CICIDS2017 Thursday (Working Hours) — All Benign Traffic
==============================================================================
Evaluates XGBoost FP rate on actual problem data: CICIDS2017 Thursday WorkingHours
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
    X_scaled = scaler.transform(X_benign)
    probs = model.predict_proba(X_scaled)
    prob_atk = 1.0 - probs[:, 0]
    preds = (prob_atk >= threshold).astype(int)
    return preds.mean(), preds


def run_coral_on_cicids2017_thursday():
    print("=" * 70)
    print("CORAL EVALUATION — CICIDS2017 Thursday WorkingHours (ALL BENIGN)")
    print("=" * 70)
    
    model_path = "./ids_model_v7_final.pkl"
    source_eve = "./eve_labeled/full_dataset/eve.json"
    target_eve = "./eve_labeled/cicids2017_thursday_eve.json"
    
    # Load model
    print("\n[1/5] Loading XGBoost v7 model...")
    model, scaler, threshold, inv_class_map = load_model(model_path)
    print(f"    Threshold (from model bundle): {threshold}")
    print(f"    Classes: {inv_class_map}")
    
    # Extract source features (training distribution)
    print("\n[2/5] Extracting source features (CICIDS2018 full_dataset, 50k)...")
    X_source, _ = fe.load_eve_features(source_eve, max_samples=50000)
    print(f"    Source shape: {X_source.shape}")
    
    # Extract target features (CICIDS2017 Thursday - ALL BENIGN)
    print("\n[3/5] Extracting target features (CICIDS2017 Thursday)...")
    X_target, _ = fe.load_eve_features(target_eve, max_samples=50000)
    print(f"    Target shape: {X_target.shape}")
    print(f"    Assuming ALL flows are BENIGN (WorkingHours traffic)")
    
    # BEFORE CORAL
    print("\n[4/5] Evaluating XGBoost FP rate BEFORE CORAL...")
    fp_before, preds_before = evaluate_fp_rate(model, scaler, threshold, X_target)
    print(f"    FP Rate (before): {fp_before:.4f} ({fp_before*100:.2f}%)")
    print(f"    FP Count: {int(fp_before*len(X_target))} / {len(X_target)}")
    
    # CORAL Adaptation
    print("\n[5/5] Fitting CORAL adapter...")
    from coral_domain_adaptation import CORALDomainAdapter
    adapter = CORALDomainAdapter(lambda_reg=1e-5)
    adapter.fit(X_source, X_target, scale=True)
    metrics = adapter.get_metrics()
    print(f"    Frobenius ||Cs-Ct||_F: {metrics['frobenius_distance']:.6f}")
    print(f"    Trace ratio (Ct/Cs): {metrics['trace_ratio_target_over_source']:.6f}")
    print(f"    Cond. number (source): {metrics['condition_number_source']:.2f}")
    print(f"    Cond. number (target): {metrics['condition_number_target']:.2f}")
    
    # Transform target to source space
    X_target_aligned = adapter.transform_target(X_target)
    
    # AFTER CORAL
    print("\nEvaluating XGBoost FP rate AFTER CORAL alignment...")
    fp_after, preds_after = evaluate_fp_rate(model, scaler, threshold, X_target_aligned)
    print(f"    FP Rate (after): {fp_after:.4f} ({fp_after*100:.2f}%)")
    print(f"    FP Count: {int(fp_after*len(X_target))} / {len(X_target)}")
    
    # Results table
    improvement = (fp_before - fp_after) / fp_before * 100 if fp_before > 0 else 0
    fp_count_before = int(fp_before * len(X_target))
    fp_count_after = int(fp_after * len(X_target))
    
    print("\n" + "=" * 70)
    print("RESULTS: CICIDS2017 Thursday WorkingHours — BEFORE vs AFTER CORAL")
    print("=" * 70)
    print(f"{'Metric':<38} {'Before':>12} {'After':>12} {'Improvement':>12}")
    print("-" * 70)
    print(f"{'FP Rate (All Benign target)':<38} {fp_before:>11.4f} {fp_after:>11.4f} {improvement:>11.1f}%")
    print(f"{'FP Count':<38} {fp_count_before:>12d} {fp_count_after:>12d} {fp_count_before - fp_count_after:>11d}")
    print(f"{'Frobenius ||Cs-Ct||_F':<38} {'N/A':>12} {metrics['frobenius_distance']:>12.6f} {'-':>12}")
    print(f"{'Trace Ratio (Ct/Cs)':<38} {'N/A':>12} {metrics['trace_ratio_target_over_source']:>12.6f} {'-':>12}")
    print(f"{'Condition Number (Source)':<38} {'N/A':>12} {metrics['condition_number_source']:>12.2f} {'-':>12}")
    print(f"{'Condition Number (Target)':<38} {'N/A':>12} {metrics['condition_number_target']:>12.2f} {'-':>12}")
    print("=" * 70)
    
    results = {
        'dataset': 'CICIDS2017_Thursday_WorkingHours',
        'fp_before': float(fp_before),
        'fp_after': float(fp_after),
        'improvement_pct': float(improvement),
        'fp_count_before': fp_count_before,
        'fp_count_after': fp_count_after,
        'frobenius': float(metrics['frobenius_distance']),
        'trace_ratio': float(metrics['trace_ratio_target_over_source']),
        'cond_source': float(metrics['condition_number_source']),
        'cond_target': float(metrics['condition_number_target']),
        'n_target': int(len(X_target)),
        'n_source': int(len(X_source)),
        'threshold': float(threshold),
    }
    
    with open('./coral_eval_cicids2017_thursday.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to coral_eval_cicids2017_thursday.json")
    
    return results


if __name__ == '__main__':
    run_coral_on_cicids2017_thursday()