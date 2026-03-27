#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib

from baseline_utils import (
    load_data, build_feature_matrix, print_evaluation,save_results_json,
    MODELS_DIR, RANDOM_SEED, TEST_SIZE
)


def main():
    print("\n" + "=" * 60)
    print("       SVM CLASSIFIER (Support Vector Machine)")
    print("=" * 60)
    
    # Load data
    df_keypoints, df_labels, df_segments = load_data()
    
    # Build feature matrix
    X, y, feature_names, metadata = build_feature_matrix(df_keypoints, df_labels, df_segments)
    
    if len(X) < 50:
        print(f"\n❌ Insufficient data: only {len(X)} samples. Need at least 50.")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    print(f"\n   Classes: {class_names}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    print(f"\n   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Enable probability estimates
        random_state=RANDOM_SEED
    )
    
    # Cross-validation
    print("\n" + "-" * 60)
    print("  CROSS-VALIDATION (5-fold)")
    print("-" * 60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
    
    print(f"\n   CV F1 scores: {cv_scores}")
    print(f"   CV F1 mean:   {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Train on full training set
    print("\n🧠 Training SVM...")
    model.fit(X_train_scaled, y_train)
    print(f"   Support vectors: {model.n_support_}")
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    

    metrics = print_evaluation(y_test, y_pred, y_prob, class_names, "SVM")
    save_results_json(metrics, "svm")

    model_path = os.path.join(MODELS_DIR, "svm_classifier.joblib")
    scaler_path = os.path.join(MODELS_DIR, "svm_scaler.joblib")
    encoder_path = os.path.join(MODELS_DIR, "svm_label_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    
    print(f"\n💾 Model saved: {model_path}")
    print(f"💾 Scaler saved: {scaler_path}")
    print(f"💾 Label encoder saved: {encoder_path}")

    results = {
        'model': 'SVM',
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        **metrics
    }
    
    print("\n✅ Done!")
    
    return results


if __name__ == "__main__":
    main()