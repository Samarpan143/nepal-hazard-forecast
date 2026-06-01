# Chapter 5: Deep Learning and LSTMs
## Temporal Intelligence for Himalayan Hazards

### 5.1 Why LSTM for Nepal?
In a standard **Random Forest (RF)** model, the AI only sees "Today."
*   **The Problem:** If it rains 50mm today on dry soil, there might be no landslide.
*   **The Solution:** If it rains 50mm today on soil that has been saturated for 6 days, a landslide is imminent.
*   **The Model:** The **Long Short-Term Memory (LSTM)** network is designed to "remember" the previous 7 days of environment states.

### 5.2 The 3D Tensor Transformation
Standard neural networks take 2D arrays `[Samples, Features]`.
LSTMs require a 3D matrix `[Samples, Time_Steps, Features]`.
*   **Our Logic:** We transform our 3-year CSV into sliding windows.
*   **Window Size:** 7 days.
*   **Input Shape:** `(Days, 7, 5_Sensors)`.
*   **Output:** The binary classification `Hazard_Risk (0 or 1)` for the 8th day.

### 5.3 The Model Architecture
We implemented the LSTM in **TensorFlow/Keras**:
```python
model = Sequential([
    LSTM(64, input_shape=(7, 10)), # 64 hidden units
    Dropout(0.2), # Prevent overfitting
    Dense(32, activation='relu'), # Deep layer
    Dense(1, activation='sigmoid') # Binary output (0 to 1)
])
```
*   **Activation:** We use the `Sigmoid` function at the end so the model outputs a **Probability**. 0.85 means there is an 85% probability of hazard risk.

### 5.4 The Ensemble Strategy
To stabilize the results, we use a **Voting Classifier**.
We combine the "Static Intelligence" of a Random Forest with the "Temporal Intelligence" of the LSTM. If both models agree, the risk alert is highly confident.

---
*Next Chapter: [Chapter 6: Verification and Case Studies](06_Verification_and_Case_Studies.md)*
