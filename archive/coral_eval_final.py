#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORAL Evaluation — Production Threshold (0.66) + CORAL on CICIDS2017 Thursday
===============================================================================
Uses canonical coral_domain_adaptation.py from pipeline/
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, "./pipeline")

from coral_domain_adaptation import CORALDomainAdapter, load_unlabeled_streams


def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['model'], bundle['scaler'], bundle.get('threshold', 0.66), bundle.get('inv_class_map', {})


def evaluate_fp_rate(model, scaler, threshold, X_benign):
    """Evaluate false positive rate on benign traffic."""
    X_scaled = scaler.transform(X_benign)
    probs = model.predict_proba(X_scaled)
    prob_atk = 1.0 - probs[:, 0]
    preds = (prob_atk >= threshold).astype(int)
    return preds.mean(), preds


def main():
    print("=" * 70)
    print("CORAL EVALUATION — PRODUCTION THRESHOLD (0.66) + CORAL")
    print("Dataset: CICIDS2017 Thursday WorkingHours (ALL BENIGN)")
    print("=" * 70)

    # Paths
    model_path = "./models/ids_model_v7_final.pkl"
    source_eve = "./data/eve/full_dataset/eve.json"
    # Use the Suricata-processed Thursday pcap output
    thursday_eve = "./data/eve/cicids2017_thursday_eve.json"

    # Load model
    print("\n[1/5] Loading XGBoost v7 model...")
    model, scaler, pickle_threshold, inv_class_map = load_model(model_path)
    print(f"    Pickle threshold (training-optimized): {pickle_threshold}")
    print(f"    Production threshold (TEKNIK_SPEC.md): 0.66")
    print(f"    Classes: {inv_class_map}")

    # Production threshold per TEKNIK_SPEC.md
    PRODUCTION_THRESHOLD = 0.66

    # Extract source features (training distribution)
    print("\n[2/5] Extracting source features (50k from full_dataset)...")
    X_source, _ = load_unlabeled_streams(source_eve, max_samples=50000)
    print(f"    Source shape: {X_source.shape}")

    # Extract target features (CICIDS2017 Thursday - all benign)
    print("\n[3/5] Extracting target features (Thursday WorkingHours)...")
    X_target, _ = load_unlabeled_streams(thursday_eve)
    print(f"    Target shape: {X_target.shape}")
    print(f"    All benign (CICIDS2017 Thursday = working hours benign traffic)")

    # BEFORE CORAL with production threshold
    print(f"\n[4/5] Evaluating XGBoost FP rate BEFORE CORAL (threshold={PRODUCTION_THRESHOLD})...")
    fp_before, preds_before = evaluate_fp_rate(model, scaler, PRODUCTION_THRESHOLD, X_target)
    print(f"    FP Rate (before): {fp_before:.4f} ({fp_before*100:.2f}%)")
    print(f"    FP Count: {int(fp_before*len(X_target))} / {len(X_target)}")

    # CORAL Adaptation
    print("\n[5/5] Fitting CORAL adapter...")
    adapter = CORALDomainAdapter(lambda_reg=1e-5)
    adapter.fit(X_source, X_target, scale=True)
    metrics = adapter.get_metrics()
    print(f"    Frobenius ||Cs-Ct||_F: {metrics['frobenius_distance']:.6f}")
    print(f"    Trace ratio (Ct/Cs): {metrics['trace_ratio_target_over_source']:.6f}")
    print(f"    Cond. number (source): {metrics['condition_number_source']:.2f}")
    print(f"    Cond. number (target): {metrics['condition_number_target']:.2f}")

    # Transform target to source space
    X_target_aligned = adapter.transform_target(X_target)

    # AFTER CORAL with production threshold
    print(f"\nEvaluating XGBoost FP rate AFTER CORAL (threshold={PRODUCTION_THRESHOLD})...")
    fp_after, preds_after = evaluate_fp_rate(model, scaler, PRODUCTION_THRESHOLD, X_target_aligned)
    print(f"    FP Rate (after): {fp_after:.4f} ({fp_after*100:.2f}%)")
    print(f"    FP Count: {int(fp_after*len(X_target))} / {len(X_target)}")

    # Results table
    improvement = (fp_before - fp_after) / fp_before * 100 if fp_before > 0 else 0
    fp_count_before = int(fp_before * len(X_target))
    fp_count_after = int(fp_after * len(X_target))

    print("\n" + "=" * 70)
    print("FINAL: CICIDS2017 Thursday — PRODUCTION (0.66) + CORAL")
    print("=" * 70)
    print(f"{'Metric':<40} {'Before':>12} {'After':>12} {'Improvement':>12}")
    print("-" * 70)
    print(f"{'FP Rate @ 0.66 (All Benign target)':<40} {fp_before:>11.4f} {fp_after:>11.4f} {improvement:>11.1f}%")
    print(f"{'FP Count':<40} {fp_count_before:>12d} {fp_count_after:>12d} {fp_count_before - fp_count_after:>11d}")
    print(f"{'Frobenius ||Cs-Ct||_F':<40} {'N/A':>12} {metrics['frobenius_distance']:>12.6f} {'-':>12}")
    print(f"{'Trace Ratio (Ct/Cs)':<40} {'N/A':>12} {metrics['trace_ratio_target_over_source']:>12.6f} {'-':>12}")
    print(f"{'Condition Number (Source)':<40} {'N/A':>12} {metrics['condition_number_source']:>12.2f} {'-':>12}")
    print(f"{'Condition Number (Target)':<40} {'N/A':>12} {metrics['condition_number_target']:>12.2f} {'-':>12}")
    print("=" * 70)

    # Also show with pickle threshold for reference
    print(f"\n[Reference] With pickle threshold ({pickle_threshold}):")
    fp_before_p, _ = evaluate_fp_rate(model, scaler, pickle_threshold, X_target)
    fp_after_p, _ = evaluate_fp_rate(model, scaler, pickle_threshold, X_target_aligned)
    print(f"    Before CORAL: {fp_before_p:.4f} ({fp_before_p*100:.2f}%)")
    print(f"    After CORAL:  {fp_after_p:.4f} ({fp_after_p*100:.2f}%)")

    # Save results
    results = {
        'dataset': 'CICIDS2017_Thursday_WorkingHours',
        'production_threshold': PRODUCTION_THRESHOLD,
        'pickle_threshold': float(pickle_threshold),
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
    }

    output_path = "./coral_eval_final.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()