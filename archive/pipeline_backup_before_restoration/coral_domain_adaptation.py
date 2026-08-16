#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORAL Domain Adaptation Implementation for Hybrid IDS
======================================================
DeepSeek Report Step 3: Aligning 50k unlabeled streams to source distribution
via Correlation Alignment (CORAL) - matching second-order statistics.

Reference: Sun & Saenko, "Correlation Alignment for Unsupervised Domain Adaptation" (2016)
DeepCORAL: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (2016)
adapt-python library implementation reference

Integration: Neomotron3 codebase reference spec (hybrid_inference.py, features.py)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted
import warnings
warnings.filterwarnings("ignore")


class CORAL(BaseEstimator, TransformerMixin):
    """
    CORrelation ALignment (CORAL) for Unsupervised Domain Adaptation.
    
    Aligns second-order statistics (covariance) of source and target domains
    via closed-form linear transformation. No target labels required.
    
    Transformation: X_s_aligned = X_s * C_s^(-1/2) * C_t^(1/2)
    where C_s, C_t are regularized covariance matrices of source and target.
    
    Parameters
    ----------
    lambda_reg : float, default=1e-5
        Regularization parameter for covariance matrix inversion.
        Larger values = less adaptation (more conservative).
    copy : bool, default=True
        Whether to copy input data or transform in-place.
    """
    
    def __init__(self, lambda_reg: float = 1e-5, copy: bool = True):
        self.lambda_reg = lambda_reg
        self.copy = copy
        self.Cs_ = None
        self.Ct_ = None
        self.transform_matrix_ = None
        self.source_mean_ = None
        self.target_mean_ = None
        self.fitted_ = False
    
    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute regularized covariance matrix."""
        n_samples, n_features = X.shape
        cov = np.cov(X.T) + self.lambda_reg * np.eye(n_features)
        return cov
    
    def _matrix_sqrt_inv(self, C: np.ndarray) -> np.ndarray:
        """Compute C^(-1/2) via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    def _matrix_sqrt(self, C: np.ndarray) -> np.ndarray:
        """Compute C^(1/2) via eigendecomposition."""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    def fit(self, Xs: np.ndarray, Xt: np.ndarray) -> 'CORAL':
        """
        Fit CORAL transformation from source to target domain.
        
        Parameters
        ----------
        Xs : array-like, shape (n_samples_s, n_features)
            Source domain features (labeled training data).
        Xt : array-like, shape (n_samples_t, n_features)
            Target domain features (unlabeled, e.g., 50k unlabeled streams).
            
        Returns
        -------
        self : CORAL
            Fitted transformer.
        """
        Xs = check_array(Xs, dtype=np.float64, copy=self.copy)
        Xt = check_array(Xt, dtype=np.float64, copy=self.copy)
        
        if Xs.shape[1] != Xt.shape[1]:
            raise ValueError(f"Feature dimension mismatch: source {Xs.shape[1]} vs target {Xt.shape[1]}")
        
        # Center data
        self.source_mean_ = Xs.mean(axis=0)
        self.target_mean_ = Xt.mean(axis=0)
        Xs_centered = Xs - self.source_mean_
        Xt_centered = Xt - self.target_mean_
        
        # Compute regularized covariance matrices
        self.Cs_ = self._compute_covariance(Xs_centered)
        self.Ct_ = self._compute_covariance(Xt_centered)
        
        # Compute transformation matrix: C_s^(-1/2) * C_t^(1/2)
        Cs_sqrt_inv = self._matrix_sqrt_inv(self.Cs_)
        Ct_sqrt = self._matrix_sqrt(self.Ct_)
        self.transform_matrix_ = Cs_sqrt_inv @ Ct_sqrt
        
        self.fitted_ = True
        return self
    
    def transform(self, X: np.ndarray, domain: str = 'source') -> np.ndarray:
        """
        Transform features using CORAL alignment.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to transform.
        domain : str, default='source'
            'source' -> align source to target (for training)
            'target' -> align target to source (for inference on target domain)
            
        Returns
        -------
        X_aligned : array-like, shape (n_samples, n_features)
            Aligned features.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X, dtype=np.float64, copy=self.copy)
        
        if domain == 'source':
            # Source -> Target alignment (whiten source, color to target)
            X_centered = X - self.source_mean_
            X_aligned = X_centered @ self.transform_matrix_
            X_aligned = X_aligned + self.target_mean_
        elif domain == 'target':
            # Target -> Source alignment (inverse transform)
            inv_transform = np.linalg.inv(self.transform_matrix_)
            X_centered = X - self.target_mean_
            X_aligned = X_centered @ inv_transform
            X_aligned = X_aligned + self.source_mean_
        else:
            raise ValueError("domain must be 'source' or 'target'")
        
        return X_aligned.astype(np.float32)
    
    def fit_transform(self, Xs: np.ndarray, Xt: np.ndarray, X: Optional[np.ndarray] = None, 
                      domain: str = 'source') -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(Xs, Xt)
        if X is None:
            X = Xs
        return self.transform(X, domain=domain)
    
    def coral_loss(self, Xs: np.ndarray, Xt: np.ndarray) -> float:
        """
        Compute CORAL loss (Frobenius norm of covariance difference).
        Used for monitoring adaptation quality.
        """
        Xs = check_array(Xs, dtype=np.float64)
        Xt = check_array(Xt, dtype=np.float64)
        Cs = self._compute_covariance(Xs - Xs.mean(axis=0))
        Ct = self._compute_covariance(Xt - Xt.mean(axis=0))
        diff = Cs - Ct
        return float(np.sum(diff ** 2))
    
    def get_alignment_quality(self) -> Dict[str, float]:
        """Return alignment quality metrics."""
        check_is_fitted(self, 'fitted_')
        cov_diff = self.Cs_ - self.Ct_
        frobenius_norm = float(np.linalg.norm(cov_diff, 'fro'))
        trace_ratio = float(np.trace(self.Ct_) / np.trace(self.Cs_))
        return {
            'frobenius_distance': frobenius_norm,
            'trace_ratio_target_over_source': trace_ratio,
            'condition_number_source': float(np.linalg.cond(self.Cs_)),
            'condition_number_target': float(np.linalg.cond(self.Ct_)),
        }


class DeepCORAL:
    """
    Deep CORAL loss for neural network training (e.g., LSTM).
    
    Adds CORAL loss as regularization term to align hidden layer
    activations between source and target domains during training.
    
    Loss = L_task + lambda * L_coral
    L_coral = 1/(4*d^2) * ||C_s - C_t||_F^2
    
    Reference: Sun & Saenko, "Deep CORAL" (ECCV 2016)
    """
    
    def __init__(self, lambda_coral: float = 1.0, layer_name: str = 'lstm_1'):
        self.lambda_coral = lambda_coral
        self.layer_name = layer_name
    
    @staticmethod
    def coral_loss(source_features: np.ndarray, target_features: np.ndarray) -> float:
        """
        Compute CORAL loss between source and target feature activations.
        
        Parameters
        ----------
        source_features : array-like, shape (batch_size, feature_dim)
        target_features : array-like, shape (batch_size, feature_dim)
            
        Returns
        -------
        loss : float
            CORAL loss value.
        """
        def _cov(x):
            x_centered = x - x.mean(axis=0, keepdims=True)
            return (x_centered.T @ x_centered) / (x.shape[0] - 1)
        
        Cs = _cov(source_features)
        Ct = _cov(target_features)
        d = source_features.shape[1]
        loss = np.sum((Cs - Ct) ** 2) / (4 * d * d)
        return float(loss)
    
    def compute_loss(self, source_activations: np.ndarray, 
                     target_activations: np.ndarray) -> float:
        """Compute weighted CORAL loss for training."""
        return self.lambda_coral * self.coral_loss(source_activations, target_activations)


class CORALDomainAdapter:
    """
    High-level CORAL domain adapter for IDS hybrid system.
    
    Integrates with existing pipeline:
    - Source: Labeled CICIDS2018 + MAWI + CTU-13 + MCFP (XGBoost/LSTM training data)
    - Target: 50k unlabeled live streams (production traffic)
    
    Usage:
        adapter = CORALDomainAdapter(lambda_reg=1e-5)
        adapter.fit(X_source_labeled, X_target_unlabeled_50k)
        X_source_aligned = adapter.transform(X_source_labeled, domain='source')
        # Train XGBoost/LSTM on X_source_aligned
        # At inference: X_live_aligned = adapter.transform(X_live, domain='target')
    """
    
    def __init__(self, lambda_reg: float = 1e-5, scaler: Optional[StandardScaler] = None):
        self.coral = CORAL(lambda_reg=lambda_reg)
        self.scaler = scaler or StandardScaler()
        self.is_fitted_ = False
        self.n_features_ = None
        self.alignment_metrics_ = {}
    
    def fit(self, X_source: np.ndarray, X_target: np.ndarray,
            scale: bool = True) -> 'CORALDomainAdapter':
        """
        Fit CORAL adapter on source (labeled) and target (unlabeled) data.
        
        Parameters
        ----------
        X_source : array-like, shape (n_source_samples, n_features)
            Labeled source domain features (e.g., training data from CICIDS2018+MAWI+CTU+MCFP)
        X_target : array-like, shape (n_target_samples, n_features)
            Unlabeled target domain features (e.g., 50k live eve.json streams)
        scale : bool, default=True
            Whether to standardize features before CORAL alignment.
            
        Returns
        -------
        self : CORALDomainAdapter
        """
        X_source = check_array(X_source, dtype=np.float64)
        X_target = check_array(X_target, dtype=np.float64)
        
        self.n_features_ = X_source.shape[1]
        
        if X_target.shape[1] != self.n_features_:
            raise ValueError(f"Feature mismatch: source={self.n_features_}, target={X_target.shape[1]}")
        
        # Scale features (important for covariance stability)
        if scale:
            X_source_scaled = self.scaler.fit_transform(X_source)
            X_target_scaled = self.scaler.transform(X_target)
        else:
            X_source_scaled = X_source
            X_target_scaled = X_target
        
        # Fit CORAL
        self.coral.fit(X_source_scaled, X_target_scaled)
        self.is_fitted_ = True
        self.alignment_metrics_ = self.coral.get_alignment_quality()
        
        return self
    
    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        """Transform source features to align with target domain (for training)."""
        if not self.is_fitted_:
            raise RuntimeError("Adapter not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X_source)
        return self.coral.transform(X_scaled, domain='source')
    
    def transform_target(self, X_target: np.ndarray) -> np.ndarray:
        """Transform target features to align with source domain (for inference)."""
        if not self.is_fitted_:
            raise RuntimeError("Adapter not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X_target)
        return self.coral.transform(X_scaled, domain='target')
    
    def transform(self, X: np.ndarray, domain: str = 'target') -> np.ndarray:
        """Generic transform."""
        if domain == 'source':
            return self.transform_source(X)
        elif domain == 'target':
            return self.transform_target(X)
        else:
            raise ValueError("domain must be 'source' or 'target'")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return alignment quality metrics."""
        if not self.is_fitted_:
            return {}
        return {
            **self.alignment_metrics_,
            'n_features': self.n_features_,
            'lambda_reg': self.coral.lambda_reg,
        }
    
    def save(self, path: str) -> None:
        """Save fitted adapter to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted adapter.")
        with open(path, 'wb') as f:
            pickle.dump({
                'coral': self.coral,
                'scaler': self.scaler,
                'is_fitted_': self.is_fitted_,
                'n_features_': self.n_features_,
                'alignment_metrics_': self.alignment_metrics_,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CORALDomainAdapter':
        """Load fitted adapter from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        adapter = cls()
        adapter.coral = data['coral']
        adapter.scaler = data['scaler']
        adapter.is_fitted_ = data['is_fitted_']
        adapter.n_features_ = data['n_features_']
        adapter.alignment_metrics_ = data['alignment_metrics_']
        return adapter


def load_unlabeled_streams(eve_path: str, max_samples: int = 50000,
                           feature_extractor=None) -> np.ndarray:
    """
    Load 50k unlabeled streams from eve.json for target domain.
    
    Parameters
    ----------
    eve_path : str
        Path to eve.json file (live Suricata output).
    max_samples : int, default=50000
        Maximum number of flows to extract.
    feature_extractor : callable, optional
        Function to extract features from flow event. 
        If None, uses extract_features_v7 from features.py.
        
    Returns
    -------
    X_target : np.ndarray, shape (n_samples, n_features)
        Unlabeled target domain features.
    """
    import json
    from pathlib import Path
    
    if feature_extractor is None:
        try:
            from model_eğitim_dosyaları.features import extract_features_v7
            feature_extractor = extract_features_v7
        except ImportError:
            raise ImportError("features.py not found. Provide feature_extractor function.")
    
    X_list = []
    count = 0
    
    with open(eve_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get('event_type') == 'flow':
                    feat = feature_extractor(event)
                    if feat is not None:
                        X_list.append(feat)
                        count += 1
            except json.JSONDecodeError:
                continue
    
    if not X_list:
        raise ValueError(f"No valid flow events found in {eve_path}")
    
    return np.stack(X_list).astype(np.float32)


def run_coral_adaptation_step3(source_features_path: str,
                                target_eve_path: str,
                                output_adapter_path: str,
                                max_target_samples: int = 50000,
                                lambda_reg: float = 1e-5) -> Dict[str, Any]:
    """
    DeepSeek Report Step 3: CORAL Domain Adaptation
    
    Align 50k unlabeled target streams to source distribution.
    
    Pipeline:
    1. Load source features (labeled training data: XGBoost/LSTM features)
    2. Load 50k unlabeled target streams from live eve.json
    3. Fit CORAL adapter (covariance alignment)
    4. Save adapter for inference pipeline integration
    5. Return alignment quality metrics
    
    Parameters
    ----------
    source_features_path : str
        Path to source features .npy or .pkl (labeled training data, 70 or 78 features)
    target_eve_path : str
        Path to live eve.json for unlabeled target streams
    output_adapter_path : str
        Path to save fitted CORAL adapter (.pkl)
    max_target_samples : int, default=50000
        Number of unlabeled streams to use for target covariance estimation
    lambda_reg : float, default=1e-5
        CORAL regularization parameter
        
    Returns
    -------
    metrics : dict
        Alignment quality metrics (Frobenius distance, trace ratio, condition numbers)
    """
    print("=" * 60)
    print("CORAL DOMAIN ADAPTATION - DeepSeek Step 3")
    print("Aligning 50k unlabeled streams to source distribution")
    print("=" * 60)
    
    # Load source features (labeled training data)
    print(f"\n[1/4] Loading source features from {source_features_path}...")
    if source_features_path.endswith('.npy'):
        X_source = np.load(source_features_path)
    elif source_features_path.endswith('.pkl'):
        with open(source_features_path, 'rb') as f:
            data = pickle.load(f)
            X_source = data['features'] if isinstance(data, dict) and 'features' in data else data
    else:
        raise ValueError("Source features must be .npy or .pkl")
    print(f"    Source shape: {X_source.shape}")
    
    # Load 50k unlabeled target streams
    print(f"\n[2/4] Loading {max_target_samples} unlabeled target streams from {target_eve_path}...")
    X_target = load_unlabeled_streams(target_eve_path, max_samples=max_target_samples)
    print(f"    Target shape: {X_target.shape}")
    
    # Fit CORAL adapter
    print(f"\n[3/4] Fitting CORAL adapter (lambda={lambda_reg})...")
    adapter = CORALDomainAdapter(lambda_reg=lambda_reg)
    adapter.fit(X_source, X_target, scale=True)
    
    metrics = adapter.get_metrics()
    print(f"    Alignment metrics:")
    for k, v in metrics.items():
        print(f"      {k}: {v:.6f}" if isinstance(v, float) else f"      {k}: {v}")
    
    # Save adapter
    print(f"\n[4/4] Saving adapter to {output_adapter_path}...")
    adapter.save(output_adapter_path)
    print("    Done.")
    
    print("\n" + "=" * 60)
    print("CORAL ADAPTATION COMPLETE")
    print("Adapter ready for integration with hybrid_inference.py")
    print("=" * 60)
    
    return metrics


def integrate_with_hybrid_inference(adapter_path: str,
                                     hybrid_inference_path: str = 'model_eğitim_dosyaları/hybrid_inference.py'):
    """
    Generate integration code snippet for hybrid_inference.py.
    
    Adds CORAL target->source transformation before XGBoost/LSTM inference.
    """
    integration_code = f'''
# ============================================================
# CORAL DOMAIN ADAPTATION INTEGRATION (DeepSeek Step 3)
# Insert into hybrid_inference.py after feature extraction
# ============================================================

from coral_domain_adaptation import CORALDomainAdapter
import numpy as np

# Load fitted CORAL adapter (target -> source alignment for inference)
coral_adapter = CORALDomainAdapter.load(r"{adapter_path}")

# In process_event() or batch inference, after extract_features_v7():
# X_raw = extract_features_v7(event)  # shape: (70,) or (78,)
# X_scaled = scaler.transform(X_raw.reshape(1, -1))  # existing scaler
# X_aligned = coral_adapter.transform_target(X_scaled)  # NEW: align to source
# 
# Then use X_aligned for XGBoost/LSTM inference instead of X_scaled

# For LSTM window (40 flows x 78 features):
# window_features = np.array([coral_adapter.transform_target(scaler.transform(f.reshape(1,-1)))[0] 
#                              for f in flow_window])  # shape: (40, 78)
'''
    return integration_code


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CORAL Domain Adaptation - DeepSeek Step 3')
    parser.add_argument('--source', required=True, help='Source features (.npy or .pkl)')
    parser.add_argument('--target-eve', required=True, help='Target eve.json (unlabeled streams)')
    parser.add_argument('--output', required=True, help='Output adapter path (.pkl)')
    parser.add_argument('--max-target', type=int, default=50000, help='Max target samples')
    parser.add_argument('--lambda-reg', type=float, default=1e-5, help='CORAL regularization')
    
    args = parser.parse_args()
    
    run_coral_adaptation_step3(
        source_features_path=args.source,
        target_eve_path=args.target_eve,
        output_adapter_path=args.output,
        max_target_samples=args.max_target,
        lambda_reg=args.lambda_reg
    )