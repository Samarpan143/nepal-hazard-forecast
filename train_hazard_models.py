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
    
    # Map text labels to integers (0=Stable, 1=Landslide, 2=Flood, 3=Fire)
    df_raw = pd.read_csv('sample_locations_raw.csv')
    label_map = {
        'Stable': 0,
        'Landslide': 1,
        'Flood': 2,
        'Forest Fire': 3
    }
    df['multiclass_label'] = df['sample_id'].map(df_raw['Hazard']).map(label_map).fillna(0).astype(int)
    
    time_steps = 14
    # The 11 environmental features we extract for each day
    base_features = ['VV', 'VH', 'NDVI', 'NDMI', 'Aerosol', 'LST', 'pr', 'soil', 'Slope', 'Aspect', 'TPI']
    
    # Reshape the flat CSV data into 3D sequences for the LSTM
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
        labels.append(row['multiclass_label'])
        
    X = np.array(samples)
    y = np.array(labels)
    
    print(f"Data Shape: {X.shape}, Label Shape: {y.shape}")
    print("Class Counts:")
    for label_val, label_name in [(0, 'Stable'), (1, 'Landslide'), (2, 'Flood'), (3, 'Forest Fire')]:
        print(f"  {label_name}: {np.sum(y == label_val)}")
    
    # Standardize features so they are all on the same scale
    num_samples, ts, num_feats = X.shape
    X_flat = X.reshape(-1, num_feats)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(num_samples, ts, num_feats)
    
    # Split 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Tree models like Random Forest need 2D data, not 3D sequences
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Calculate base class weights to handle natural class imbalance
    counts = np.bincount(y_train)
    total = sum(counts)
    class_weight = {i: (1 / counts[i]) * (total / len(counts)) for i in range(len(counts))}
    
    # Artificially boost the importance of floods and fires so the model doesn't ignore them
    if 2 in class_weight:
        class_weight[2] *= 2.5  # Boost flood importance by 2.5x
    if 3 in class_weight:
        class_weight[3] *= 2.0  # Boost fire importance by 2.0x
        
    print(f"Adjusted Class Weights for High Recall: {class_weight}")
    
    # Dictionary to keep track of how each model performs
    results = {}
    
    # Train Baseline 1: Random Forest
    print("\nBASELINE 1: Random Forest Classifier")
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
    
    # Train Baseline 2: Gradient Boosting
    print("\nBASELINE 2: Gradient Boosting (XGBoost-style)")
    
    # Attempt to use XGBoost, but fall back to Sklearn if not installed
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
    
    # Train Baseline 3: Standard LSTM
    print("\nBASELINE 3: Unidirectional LSTM")
    
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
    
    # Train Primary Model: Bi-Directional LSTM
    print("\nPRIMARY: Bi-Directional LSTM (with Focal Loss)")
    
    # Custom focal loss function to penalize easy predictions and focus on hard mistakes
    def focal_loss(gamma=2.0, alpha=0.25):
        def sparse_focal_loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            
            y_true_flat = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            y_pred_flat = tf.reshape(y_pred, [-1, 4])
            
            y_true_one_hot = tf.one_hot(y_true_flat, depth=4)
            p_t = tf.reduce_sum(y_true_one_hot * y_pred_flat, axis=-1)
            weight = alpha * tf.pow(1.0 - p_t, gamma)
            
            loss_flat = weight * tf.reshape(ce, [-1])
            return tf.reshape(loss_flat, tf.shape(y_true))
        return sparse_focal_loss
    
    model = Sequential([
        Input(shape=(ts, num_feats)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
    
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
    
    # Print the final comparison table
    print("\nMODEL COMPARISON TABLE")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f}")
    
    # Save metrics to JSON for later use
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nComparison saved to model_comparison_results.json")
    
    # Plot and save the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Bi-LSTM Nepal Hazard Pipeline (500m)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_500m.png')
    
    # Export the trained Bi-LSTM weights
    model.save('nepal_hazard_model_500m.h5')
    print("Bi-LSTM model and plots saved.")

if __name__ == "__main__":
    import os
    csv_file = 'nepal_500m_sequences_flat.csv'
    if os.path.exists(csv_file):
        train_bilstm_model(csv_file)
    else:
        print(f"Error: {csv_file} not found. Wait for extraction to finish.")
