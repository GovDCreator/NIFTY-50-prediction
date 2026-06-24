import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# --- New Imports for Deep Learning ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration (Update File Paths) ---
# **CRITICAL:** Update these file names to match your data.
FILE_PATHS = ['NIFTY 50-11-10-2020-to-11-10-2021.csv','NIFTY 50-11-10-2021-to-11-10-2022.csv','NIFTY 50-11-10-2022-to-11-10-2023.csv','NIFTY 50-11-10-2023-to-11-10-2024.csv','NIFTY 50-11-10-2024-to-11-10-2025.csv'] 
TARGET_COLUMN = 'Close'
N_ITER = 30     # Number of parameter settings sampled (your requested 30 models)
CV_SPLITS = 5   # Number of splits for TimeSeriesSplit CV
N_JOBS = -1     # Use all CPU cores
LSTM_LOOK_BACK = 5 # Number of past days (timesteps) the LSTM looks back
LSTM_EPOCHS = 50 # Number of training epochs for LSTM

# --- New Function for LSTM Data Preparation (Includes FIX for dtype Error) ---
def create_lstm_sequences(X, y, look_back=5):
    """
    Converts data into sequences for LSTM training.
    Includes explicit casting to float32 to prevent 'Invalid dtype: object' error.
    """
    
    # --- FIX: Explicitly cast to float32 to resolve 'object' dtype error ---
    if isinstance(X, pd.DataFrame):
        X_data = X.values.astype('float32')
    else:
        X_data = X.astype('float32')

    if isinstance(y, pd.Series):
        y_data = y.values.astype('float32')
    else:
        y_data = y.astype('float32')
    # ----------------------------------------------------------------------
        
    # Use TimeseriesGenerator to efficiently create sequences
    generator = TimeseriesGenerator(X_data, y_data, length=look_back, batch_size=1)
    
    # Extract sequences and targets
    X_seq, y_seq = [], []
    for i in range(len(generator)):
        X_seq.append(generator[i][0][0])
        y_seq.append(generator[i][1][0])
        
    # Convert lists to final NumPy arrays
    X_seq = np.array(X_seq) 
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq, look_back


# --- 1. Data Consolidation and Feature Engineering Function (Original) ---

def create_and_process_features(df):
    """Performs all required feature engineering on the DataFrame."""
    
    # Ensure 'Date' is datetime and sort data chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    df.dropna(inplace=True)
    
    # --- A. Date Features (Cyclical and One-Hot) ---
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Cyclical Encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 366)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 366)

    # One-Hot Encoding (Day of Week)
    dow_dummies = pd.get_dummies(df['DayOfWeek'], prefix='Is')
    dow_dummies = dow_dummies.rename(columns={
        'Is_0': 'Is_Monday', 'Is_1': 'Is_Tuesday', 'Is_2': 'Is_Wednesday',
        'Is_3': 'Is_Thursday', 'Is_4': 'Is_Friday'
    })
    dow_dummies.drop(columns=['Is_5', 'Is_6'], errors='ignore', inplace=True)
    df = pd.concat([df, dow_dummies], axis=1)

    # --- B. Lag and Moving Average Features ---
    df['Close_Lag1'] = df[TARGET_COLUMN].shift(1)
    df['Open_Lag1'] = df['Open'].shift(1)
    df['High_Lag1'] = df['High'].shift(1)
    df['Low_Lag1'] = df['Low'].shift(1)
    df['Close_Lag5'] = df[TARGET_COLUMN].shift(5)
    df['Close_Lag20'] = df[TARGET_COLUMN].shift(20)

    df['SMA_5'] = df[TARGET_COLUMN].rolling(window=5).mean()
    df['SMA_20'] = df[TARGET_COLUMN].rolling(window=20).mean()
    
    # Final cleaning and drop redundant time components
    df.dropna(inplace=True)
    df.drop(columns=['Date', 'Month', 'DayOfWeek', 'DayOfYear'], inplace=True, errors='ignore')
    return df.reset_index(drop=True)

# --- 2. Data Loading, Feature Creation, and Scaling (Original) ---

print("--- 1. Data Loading, Feature Creation, and Scaling ---")

all_data = []
for file in FILE_PATHS:
    try:
        all_data.append(pd.read_csv(file))
    except FileNotFoundError:
        print(f"Warning: File {file} not found. Skipping.")

if not all_data:
    print("Error: No data files found. Exiting.")
    exit()

df_combined = pd.concat(all_data, ignore_index=True)
df_processed = create_and_process_features(df_combined.copy())

y = df_processed[TARGET_COLUMN]
X = df_processed.drop(columns=[TARGET_COLUMN])

# Identify scaling features (all except binary/OHE)
binary_features = [col for col in X.columns if col.startswith('Is_')]
features_to_scale = [col for col in X.columns if col not in binary_features]

# Apply MinMaxScaler on the entire X dataset
scaler = MinMaxScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

print(f"Total rows available after processing: {len(X)}")


# --- 3. Time-Series Train/Validate/Test Split (Original) ---

print("--- 2. Custom Time-Series Splitting ---")

total_days = len(X)
# Calculate split points based on 5-year data (3.5y train, 0.5y val, 1.0y test)
train_end_idx = int(total_days * (3.5 / 5.0)) 
validation_end_idx = int(total_days * (4.0 / 5.0))

# Base Train (First 3.5 years)
X_train_base = X.iloc[:train_end_idx]
y_train_base = y.iloc[:train_end_idx]

# Validation Pool (Next 0.5 years)
X_val_pool = X.iloc[train_end_idx:validation_end_idx]
y_val_pool = y.iloc[train_end_idx:validation_end_idx]

# Final Test (Last 1 year)
X_test = X.iloc[validation_end_idx:]
y_test = y.iloc[validation_end_idx:]

print(f"Base Training size (3.5y): {len(X_train_base)}")
print(f"Validation Pool size (0.5y): {len(X_val_pool)}")
print(f"Final Testing size (1.0y): {len(X_test)}")


# --- 4. Hyperparameter Tuning using RandomizedSearchCV (Original) ---

print("\n--- 3. Hyperparameter Tuning (RandomizedSearchCV) ---")

# Define the search space using distributions
param_dist = {
    'n_estimators': randint(low=100, high=800), # Number of trees
    'max_depth': randint(low=3, high=10),      # Tree depth
    'learning_rate': uniform(loc=0.01, scale=0.2), # Learning rate (0.01 to 0.21)
    'subsample': uniform(loc=0.6, scale=0.4),  # Subsample ratio (0.6 to 1.0)
    'reg_alpha': uniform(loc=0, scale=0.5),    # L1 regularization
}

# Use TimeSeriesSplit for CV on the BASE TRAINING data
tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=N_JOBS)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=N_ITER, 
    scoring='neg_mean_squared_error', 
    cv=tscv,  # Use TimeSeriesSplit
    verbose=1,
    random_state=42,
    n_jobs=N_JOBS
)

# Fit the search on the Base Training Data (3.5 years)
random_search.fit(X_train_base, y_train_base) 

best_params = random_search.best_params_
print(f"\nBest Hyperparameters found by Randomized Search: {best_params}")

# --- 5. Final Training and Testing (UPDATED for Hybrid Model) ---

print("\n--- 4. Hybrid XGBoost + LSTM Model Training ---")

# 1. Final Training: Combine Train Base and Validation Pool
X_train_final = pd.concat([X_train_base, X_val_pool])
y_train_final = pd.concat([y_train_base, y_val_pool])

# Train the final XGBoost model using the best hyperparameters found (Used for individual and hybrid prediction)
final_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_jobs=N_JOBS,
    **best_params 
)
final_xgb_model.fit(X_train_final, y_train_final)
y_test_pred_xgb = final_xgb_model.predict(X_test)


# --- B. Prepare and Train LSTM Model ---

# Use the combined final training data for LSTM training
X_lstm_train_seq, y_lstm_train_seq, look_back = create_lstm_sequences(
    X_train_final, y_train_final, look_back=LSTM_LOOK_BACK
)

# Prepare Test Data sequences (Need to include the last 'look_back' days from train set for the first test prediction)
X_test_for_seq = pd.concat([X_train_final.iloc[-LSTM_LOOK_BACK:], X_test])
X_lstm_test_seq, _, _ = create_lstm_sequences(
    X_test_for_seq, pd.Series(np.zeros(len(X_test_for_seq))), # y-values are placeholder
    look_back=LSTM_LOOK_BACK
)

# Align the original test set (y_test and X_test_pred_xgb) to the sequence length
# The sequence creation loses the first (LSTM_LOOK_BACK - 1) days
min_len = min(len(y_test), len(X_lstm_test_seq))
X_test_aligned = X_test.iloc[-min_len:]
y_test_aligned = y_test.iloc[-min_len:]
y_test_pred_xgb_aligned = y_test_pred_xgb[-min_len:]
X_lstm_test_seq_aligned = X_lstm_test_seq[-min_len:]

# Define and Compile LSTM Model
n_features = X_lstm_train_seq.shape[2] 

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

print(f"\nTraining LSTM model (Lookback={LSTM_LOOK_BACK}, Seq_Train_Size={len(X_lstm_train_seq)})")
lstm_model.fit(
    X_lstm_train_seq, y_lstm_train_seq, 
    epochs=LSTM_EPOCHS, 
    batch_size=32, 
    verbose=0 
)

# Get LSTM Predictions on Aligned Test Data
y_test_pred_lstm_aligned = lstm_model.predict(X_lstm_test_seq_aligned).flatten()
rmse_test_lstm = np.sqrt(mean_squared_error(y_test_aligned, y_test_pred_lstm_aligned))


# --- C. Final Hybrid Prediction (Simple Averaging Ensemble) ---
# We average the two distinct model predictions for the final hybrid result.
y_test_pred_hybrid = (y_test_pred_xgb_aligned + y_test_pred_lstm_aligned) / 2
rmse_hybrid = np.sqrt(mean_squared_error(y_test_aligned, y_test_pred_hybrid))

rmse_test_xgb = np.sqrt(mean_squared_error(y_test_aligned, y_test_pred_xgb_aligned)) # Re-calculate XGB RMSE on aligned set

print(f"\nModel Evaluation (on Aligned Test Set of {min_len} rows):")
print(f"  Final Test Set RMSE (XGBoost only): {rmse_test_xgb:.2f} (Error in ₹)")
print(f"  Final Test Set R-squared (XGBoost only): {final_xgb_model.score(X_test_aligned, y_test_aligned):.4f}")
print(f"  Final Test Set RMSE (LSTM only): {rmse_test_lstm:.2f} (Error in ₹)")
print(f"  Final Test Set RMSE (Hybrid Average): **{rmse_hybrid:.2f}** (Error in ₹)")


# --- 6. Visualization of Final Test Results (UPDATED) ---

print("\n--- 5. Visualization of Final Test Results ---")

# Create a DataFrame for easy plotting
test_results_df = pd.DataFrame({
    'Actual': y_test_aligned.values,
    'XGBoost': y_test_pred_xgb_aligned,
    'LSTM': y_test_pred_lstm_aligned,
    'Hybrid (Average)': y_test_pred_hybrid
})

plt.figure(figsize=(15, 7))
plt.plot(test_results_df['Actual'], label='Actual Close Price', color='blue', linewidth=2)
plt.plot(test_results_df['XGBoost'], label='XGBoost Prediction', color='red', linestyle=':', linewidth=1)
plt.plot(test_results_df['LSTM'], label='LSTM Prediction', color='green', linestyle=':', linewidth=1)
plt.plot(test_results_df['Hybrid (Average)'], label='Hybrid (Average) Prediction', color='purple', linestyle='--', linewidth=1.5)

plt.title('Nifty 50 Close Price: Actual vs. Predictions (Test Set)')
plt.xlabel('Time (Chronological Index)')
plt.ylabel('Nifty 50 Close Price (₹)')
plt.legend()
plt.grid(True)
plt.savefig('Nifty_Hybrid_Prediction.png')
# plt.show()