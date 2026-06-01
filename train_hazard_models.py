import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

def train_bilstm_model(csv_path):
    print(f"Loading high-resolution sequences from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    time_steps = 14
    # Feature set: all 10 sensor channels extracted per timestep
    # soil = TerraClimate soil moisture (bilinear-upsampled from 4km to 500m during extraction)
    base_features = ['VV', 'VH', 'NDVI', 'NDMI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']
    
    # 1. Reshape into 3D: (Samples, 7, 10)
    samples = []
    labels = []
    
    for i, row in df.iterrows():
        sample_seq = []
        for t in range(time_steps):
            timestep_features = []
            for f in base_features:
                col_name = f"T{t}_{f}"
                val = row.get(col_name, 0)
                if pd.isna(val): val = 0
                timestep_features.append(val)
            sample_seq.append(timestep_features)
        
        samples.append(sample_seq)
        labels.append(row['label'])
        
    X = np.array(samples)
    y = np.array(labels)
    
    print(f"Data Shape: {X.shape}, Label Shape: {y.shape}")
    print(f"Positive Samples (Hazards): {sum(y)}")
    
    # 2. Scale Features
    num_samples, ts, num_feats = X.shape
    X_flat = X.reshape(-1, num_feats)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, ts, num_feats)
    
    # 3. Train/Test Split (same split for ALL models for fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Flatten 3D -> 2D for tree-based baselines (samples, 70 features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 4. Class Weights
    counts = np.bincount(y_train)
    total = sum(counts)
    class_weight = {i: (1 / counts[i]) * (total / len(counts)) for i in range(len(counts))}
    print(f"Class Weights: {class_weight}")
    
    # Store all model results for comparison
    results = {}
    
    # =========================================================================
    # BASELINE 1: Random Forest (no temporal awareness)
    # =========================================================================
    print("\n" + "="*60)
    print("BASELINE 1: Random Forest Classifier")
    print("="*60)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train_flat, y_train)
    rf_pred = rf.predict(X_test_flat)
    rf_prob = rf.predict_proba(X_test_flat)
    
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='macro'),
        'recall': recall_score(y_test, rf_pred, average='macro'),
        'f1': f1_score(y_test, rf_pred, average='macro'),
        'auc': roc_auc_score(y_test, rf_prob[:, 1]) if len(np.unique(y_test)) == 2 else roc_auc_score(y_test, rf_prob, multi_class='ovr')
    }
    print(classification_report(y_test, rf_pred))
    print(f"ROC-AUC: {results['Random Forest']['auc']:.4f}")
    
    # =========================================================================
    # BASELINE 2: XGBoost / Gradient Boosting (no temporal awareness)  
    # =========================================================================
    print("\n" + "="*60)
    print("BASELINE 2: Gradient Boosting (XGBoost-style)")
    print("="*60)
    
    # Try XGBoost if available, fall back to sklearn GradientBoosting
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', use_label_encoder=False
        )
        xgb.fit(X_train_flat, y_train)
        xgb_pred = xgb.predict(X_test_flat)
        xgb_prob = xgb.predict_proba(X_test_flat)
        model_name = 'XGBoost'
    except ImportError:
        print("XGBoost not installed, using sklearn GradientBoosting...")
        xgb = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        xgb.fit(X_train_flat, y_train)
        xgb_pred = xgb.predict(X_test_flat)
        xgb_prob = xgb.predict_proba(X_test_flat)
        model_name = 'GradientBoosting'
    
    results[model_name] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='macro'),
        'recall': recall_score(y_test, xgb_pred, average='macro'),
        'f1': f1_score(y_test, xgb_pred, average='macro'),
        'auc': roc_auc_score(y_test, xgb_prob[:, 1]) if len(np.unique(y_train)) == 2 else roc_auc_score(y_test, xgb_prob, multi_class='ovr')
    }
    print(classification_report(y_test, xgb_pred))
    print(f"ROC-AUC: {results[model_name]['auc']:.4f}")
    
    # =========================================================================
    # BASELINE 3: Unidirectional LSTM (temporal, but forward-only)
    # =========================================================================
    print("\n" + "="*60)
    print("BASELINE 3: Unidirectional LSTM")
    print("="*60)
    
    lstm_model = Sequential([
        Input(shape=(ts, num_feats)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, class_weight=class_weight, verbose=0)
    
    lstm_prob = lstm_model.predict(X_test, verbose=0)
    lstm_pred = np.argmax(lstm_prob, axis=1)
    
    results['LSTM (Unidirectional)'] = {
        'accuracy': accuracy_score(y_test, lstm_pred),
        'precision': precision_score(y_test, lstm_pred, average='macro'),
        'recall': recall_score(y_test, lstm_pred, average='macro'),
        'f1': f1_score(y_test, lstm_pred, average='macro'),
        'auc': roc_auc_score(y_test, lstm_prob[:, 1]) if len(np.unique(y_test)) == 2 else roc_auc_score(y_test, lstm_prob, multi_class='ovr')
    }
    print(classification_report(y_test, lstm_pred))
    print(f"ROC-AUC: {results['LSTM (Unidirectional)']['auc']:.4f}")
    
    # =========================================================================
    # PRIMARY MODEL: Bi-Directional LSTM
    # =========================================================================
    print("\n" + "="*60)
    print("PRIMARY: Bi-Directional LSTM")
    print("="*60)
    
    model = Sequential([
        Input(shape=(ts, num_feats)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training Bi-LSTM...")
    history = model.fit(
        X_train, y_train, 
        epochs=30, batch_size=32,
        validation_split=0.1,
        class_weight=class_weight,
        verbose=1
    )
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    results['Bi-LSTM'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'auc': roc_auc_score(y_test, y_pred_prob[:, 1]) if len(np.unique(y_test)) == 2 else roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    }
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {results['Bi-LSTM']['auc']:.4f}")
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-"*75)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f}")
    
    # Save comparison results
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nComparison saved to model_comparison_results.json")
    
    # Confusion Matrix for Bi-LSTM
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Bi-LSTM Nepal Hazard Pipeline (500m)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_500m.png')
    
    # Save Bi-LSTM model
    model.save('nepal_hazard_model_500m.h5')
    print("Bi-LSTM model and plots saved.")

if __name__ == "__main__":
    import os
    csv_file = 'nepal_500m_sequences_flat.csv'
    if os.path.exists(csv_file):
        train_bilstm_model(csv_file)
    else:
        print(f"Error: {csv_file} not found. Wait for extraction to finish.")
