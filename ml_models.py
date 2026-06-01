import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class MultiHazardModeler:
    """
    Core Machine Learning architecture for the Nepal Multi-Hazard Prediction System.
    Handles data ingestion, sequence generation for LSTMs, and Baseline/Ensemble training.
    """
    
    def __init__(self, data_path=None, dataframe=None):
        if dataframe is not None:
            self.data = dataframe
        elif data_path:
            self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        else:
            raise ValueError("Must provide either a dataframe or a data_path.")
            
        self.scaler = StandardScaler()
        self.models = {}
        
    def create_synthetic_targets(self):
        """
        GEE Proxy Target Generator: 
        Because historical disaster databases require external CSV merging, we establish conceptual synthetic proxy 
        thresholds based on severe standard deviations of the remote sensing variables to test the ML architecture.
        """
        print("Scaffolding Proxy Target Variables for modeling...")
        
        # Landslide Proxy: High Rain + Steep average topology proxy + Sparse Vegetation
        self.data['Landslide_Risk'] = ((self.data['Precipitation'] > self.data['Precipitation'].quantile(0.85)) & 
                                       (self.data['NDVI'] < self.data['NDVI'].quantile(0.40))).astype(int)
        
        # Flood Proxy: Severe, anomalous Rainfall + saturated soil
        self.data['Flood_Risk'] = ((self.data['Precipitation'] > self.data['Precipitation'].quantile(0.95)) & 
                                   (self.data['Soil_Moisture'] > self.data['Soil_Moisture'].quantile(0.80))).astype(int)
        
        # Forest Fire Proxy: Anomalously high LST, low moisture, dry vegetation
        self.data['Fire_Risk'] = ((self.data['LST_Day_1km'] > self.data['LST_Day_1km'].quantile(0.85)) & 
                                  (self.data['Soil_Moisture'] < self.data['Soil_Moisture'].quantile(0.20))).astype(int)
                                  
        # Air Pollution Proxy: Severe Aerosol load + stagnant low wind
        self.data['Pollution_Risk'] = ((self.data['absorbing_aerosol_index'] > self.data['absorbing_aerosol_index'].quantile(0.85)) & 
                                       (self.data['Wind_Speed'] < self.data['Wind_Speed'].quantile(0.30))).astype(int)

        # Unified Multi-Hazard Target: 1 if any severe hazard is present
        self.data['Any_Hazard'] = self.data[['Landslide_Risk', 'Flood_Risk', 'Fire_Risk', 'Pollution_Risk']].max(axis=1)

    def prepare_data(self, target_col='Any_Hazard', test_size=0.2, time_steps=7):
        """
        Prepares the numerical matrices for Scikit-Learn baselines and the 3D sequences for the Keras LSTM.
        """
        print(f"Preparing data splits targeting: {target_col}")
        
        # Features: all GEE remote sensing variables minus the synthetic targets
        features = [col for col in self.data.columns if not col.endswith('_Risk') and col != 'Any_Hazard']
        
        # Forward fill any lingering NaNs from GEE API timeouts
        self.data[features] = self.data[features].fillna(method='ffill').fillna(method='bfill')
        
        X = self.data[features].values
        y = self.data[target_col].values
        
        # 1. Scale Features
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Standard 2D Train/Test Split (For Random Forest / Gradient Boosting)
        self.X_train_2d, self.X_test_2d, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, shuffle=False # Time-series cannot be randomly shuffled
        )
        
        # 3. Time-Series 3D Sequence Generation (For LSTM)
        # LSTMs require data in format: [Samples, Time_Steps, Features]
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - time_steps):
            X_seq.append(X_scaled[i : i + time_steps])
            # The target is the hazard status on the day following the sequence
            y_seq.append(y[i + time_steps])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split sequential data aligned with the 2D split timeframe
        split_idx = int(len(X_seq) * (1 - test_size))
        self.X_train_3d, self.X_test_3d = X_seq[:split_idx], X_seq[split_idx:]
        self.y_train_seq, self.y_test_seq = y_seq[:split_idx], y_seq[split_idx:]
        
        self.num_features = self.X_train_2d.shape[1]
        self.time_steps = time_steps

    def train_baselines(self):
        """
        Trains and stores the baseline models (Random Forest, Gradient Boosting).
        """
        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(self.X_train_2d, self.y_train)
        self.models['RandomForest'] = rf
        
        print("Training Gradient Boosting Classifier...")
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gb.fit(self.X_train_2d, self.y_train)
        self.models['GradientBoosting'] = gb
        
        print("Training Voting Ensemble (RF + GB)...")
        ensemble = VotingClassifier(estimators=[
            ('rf', rf), 
            ('gb', gb)
        ], voting='soft')
        ensemble.fit(self.X_train_2d, self.y_train)
        self.models['Ensemble'] = ensemble

    def build_and_train_lstm(self):
        """
        Constructs and trains the Deep Learning LSTM model for sequential hazard pattern recognition.
        """
        print("Building Deep Learning LSTM Architecture...")
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.time_steps, self.num_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid') # Binary hazard classification
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Training LSTM on historical sequence data...")
        # Note: epochs=5 is low for testing. A real thesis model will use epochs=50+ with EarlyStopping
        model.fit(self.X_train_3d, self.y_train_seq, epochs=5, batch_size=32, 
                  validation_data=(self.X_test_3d, self.y_test_seq), verbose=1)
        
        self.models['LSTM'] = model

    def evaluate_models(self):
        """
        Outputs classification reports for all models over the test set.
        """
        for name, model in self.models.items():
            print(f"\n--- {name} Performance Summary ---")
            if name == 'LSTM':
                # Generate predictions and threshold at 0.5 probability
                preds = (model.predict(self.X_test_3d) > 0.5).astype(int)
                print(classification_report(self.y_test_seq, preds))
            else:
                preds = model.predict(self.X_test_2d)
                print(classification_report(self.y_test, preds))

if __name__ == "__main__":
    # Test block defining how the class is invoked with a hypothetical Data Pipeline DataFrame
    print("This module is designed to be imported. Run `data_pipeline.py` to generate the dataset first.")
