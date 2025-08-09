# 🧠 Neural Network Model – Kaggle Playground Series S5E8

This project implements a **binary classification neural network** for the [Kaggle Playground Series - Season 5, Episode 8](https://www.kaggle.com/competitions/playground-series-s5e8) competition using **TensorFlow** and **scikit-learn**.

---

## 📌 Overview
- **Goal:** Predict target `y` for given features
- **Tech Stack:** Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn
- **Key Features:**
  - One-hot encoding for categorical variables
  - Feature scaling with `StandardScaler`
  - Dropout layers for regularization
  - Class imbalance handling using computed weights
  - AUC metric for evaluation

---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 0. Load the Data ---
try:
    # Corrected file paths for Kaggle environment
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    print("Files loaded successfully!")
except FileNotFoundError as e:
    print(f"File not found. Please ensure the data files are in the correct directory. Error: {e}")
    exit()

# --- 1. Data Preparation ---
# Separate target and features
X = train_df.drop('y', axis=1)
y = train_df['y']
test_ids = test_df['id']

# Combine for consistent processing
combined_df = pd.concat([X, test_df], ignore_index=True)

# One-Hot Encode Categorical Features
categorical_features = combined_df.select_dtypes(include=['object']).columns
combined_df = pd.get_dummies(combined_df, columns=categorical_features, drop_first=True)

# Separate back into training and testing sets
X_processed = combined_df.iloc[:len(train_df)].drop('id', axis=1)
X_test_processed = combined_df.iloc[len(train_df):].drop('id', axis=1)

# --- 2. Feature Scaling (Crucial for Neural Networks) ---
# Identify numerical columns to scale (all columns are now numeric)
numerical_cols = X_processed.columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
X_processed_scaled = scaler.fit_transform(X_processed)
X_test_processed_scaled = scaler.transform(X_test_processed)


# --- 3. Build the Neural Network Model ---
# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the model architecture
model = tf.keras.Sequential([
    # Input layer - specify the input shape
    tf.keras.layers.Input(shape=(X_processed_scaled.shape[1],)),

    # First hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Dropout for regularization

    # Second hidden layer
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    # Third hidden layer
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # Output layer - sigmoid for binary classification probability
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')] # Use AUC as a metric
)

# Print model summary
model.summary()


# --- 4. Handle Class Imbalance ---
# Calculate class weights
neg, pos = np.bincount(y)
total = neg + pos
class_weight = {0: (1 / neg) * (total / 2.0),
                1: (1 / pos) * (total / 2.0)}

print(f"\nClass weights: {class_weight}")


# --- 5. Train the Model ---
print("\nStarting Neural Network training...")
history = model.fit(
    X_processed_scaled,
    y,
    epochs=15, # Number of passes through the data
    batch_size=512,
    validation_split=0.2, # Use 20% of data for validation
    class_weight=class_weight,
    verbose=1
)
print("Training complete.")


# --- 6. Prediction and Submission ---
print("\nMaking predictions with the trained Neural Network...")
# Predict probabilities on the scaled test set
test_probabilities_nn = model.predict(X_test_processed_scaled).flatten() # flatten to get a 1D array

# Create and save the new submission file
submission_df_nn = pd.DataFrame({'id': test_ids, 'y': test_probabilities_nn})
submission_df_nn.to_csv('submission_nn.csv', index=False)

print("\nNew submission file 'submission_nn.csv' created successfully!")
print(submission_df_nn.head())
