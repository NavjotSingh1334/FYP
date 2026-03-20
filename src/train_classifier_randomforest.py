#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

from baseline_utils import (
    load_data, build_feature_matrix, print_evaluation, save_results_json,
    MODELS_DIR, RANDOM_SEED, TEST_SIZE
)


def main():
    print("\n" + "=" * 60)
    print("       RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    # Load data
    df_keypoints, df_labels, df_segments = load_data()
    
    # Build feature matrix
    X, y, feature_names, metadata = build_feature_matrix(df_keypoints, df_labels, df_segments)
    
    if len(X) < 50:
        print(f"\n Insufficient data: only {len(X)} samples. Need at least 50.")
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
    
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    
    print("\n" + "-" * 60)
    print("  CROSS-VALIDATION (5-fold)")
    print("-" * 60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    print(f"\n   CV F1 scores: {cv_scores}")
    print(f"   CV F1 mean:   {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Train on full training set
    print("\n🧠 Training Random Forest...")
    model.fit(X_train, y_train)
    print(f"   Trees: {model.n_estimators}")
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = print_evaluation(y_test, y_pred, y_prob, class_names, "RANDOM FOREST")
    save_results_json(metrics, "random forest")
    # Feature importance
    print("\n" + "-" * 60)
    print("  FEATURE IMPORTANCE")
    print("-" * 60)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n   {'Rank':<6} {'Feature':<30} {'Importance':<12}")
    print(f"   {'-'*6} {'-'*30} {'-'*12}")
    
    for i, idx in enumerate(indices[:15]):
        print(f"   {i+1:<6} {feature_names[idx]:<30} {importances[idx]:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "random_forest_classifier.joblib")
    encoder_path = os.path.join(MODELS_DIR, "random_forest_label_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
  
    feature_path = os.path.join(MODELS_DIR, "random_forest_features.joblib")
    joblib.dump(feature_names, feature_path)
    
    print(f"\n💾 Model saved: {model_path}")
    print(f"💾 Label encoder saved: {encoder_path}")
    print(f"💾 Feature names saved: {feature_path}")
    
    # Save results
    results = {
        'model': 'RandomForest',
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        **metrics
    }
    
    print("\n Done!")
    
    return results


if __name__ == "__main__":
    main()