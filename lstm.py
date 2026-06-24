import pandas as pd
import numpy as np
import warnings
import xgboost as xgb # Keep import but don't use the model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit # Keep TimeSeriesSplit for data splitting visualization
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# --- Imports for Deep Learning (Essential) ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
FILE_PATHS = ['NIFTY 50-11-10-2020-to-11-10-2021.csv','NIFTY 50-11-10-2021-to-11-10-2022.csv','NIFTY 50-11-10-2022-to-11-10-2023.csv','NIFTY 50-11-10-2023-to-11-10-2024.csv','NIFTY 50-11-10-2024-to-11-10-2025.csv'] 
TARGET_COLUMN = 'Close'
LSTM_LOOK_BACK = 5 # Number of past days (timesteps) the LSTM looks back
LSTM_EPOCHS = 50 # Number of training epochs for LSTM
N_JOBS = -1     # Retained for consistency, but not used by Keras/TensorFlow

# --- Function for LSTM Data Preparation (Fixed) ---
def create_lstm_sequences(X, y, look_back=5):
    """
    Converts data into sequences for LSTM training.
    Includes explicit casting to float32 to prevent 'Invalid dtype: object' error.
    """
    
    # Explicitly cast to float32 to resolve 'object' dtype error
    if isinstance(X, pd.DataFrame):
        X_data = X.values.astype('float32')
    else:
        X_data = X.astype('float32')

    if isinstance(y, pd.Series):
        y_data = y.values.astype('float32')
    else:
        y_data = y.astype('float32')
        
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
# Calculate split points based on 5-year data (4.5y train/val, 0.5y test)
# Since we are not doing CV/RandomizedSearch, we combine the train/val pool early.
train_end_idx = int(total_days * (4.0 / 5.0)) # Use first 4 years for training

# Final Train/Validation Pool (First 4 years)
X_train_final = X.iloc[:train_end_idx]
y_train_final = y.iloc[:train_end_idx]

# Final Test (Last 1 year)
X_test = X.iloc[train_end_idx:]
y_test = y.iloc[train_end_idx:]

print(f"Final Training size (4.0y): {len(X_train_final)}")
print(f"Final Testing size (1.0y): {len(X_test)}")


# --- 4. LSTM Model Training and Testing ---

print("\n--- 3. LSTM Model Training and Testing ---")

# --- A. Prepare Data for LSTM ---

# Use the combined final training data for LSTM training
X_lstm_train_seq, y_lstm_train_seq, look_back = create_lstm_sequences(
    X_train_final, y_train_final, look_back=LSTM_LOOK_BACK
)

# Prepare Test Data sequences (Need to include the last 'look_back' days from train set)
X_test_for_seq = pd.concat([X_train_final.iloc[-LSTM_LOOK_BACK:], X_test])
X_lstm_test_seq, _, _ = create_lstm_sequences(
    X_test_for_seq, pd.Series(np.zeros(len(X_test_for_seq))), # y-values are placeholder
    look_back=LSTM_LOOK_BACK
)

# Align the original test set (y_test) to the sequence length
# The sequence creation loses the first (LSTM_LOOK_BACK - 1) days
min_len = min(len(y_test), len(X_lstm_test_seq))
y_test_aligned = y_test.iloc[-min_len:]
X_lstm_test_seq_aligned = X_lstm_test_seq[-min_len:]


# --- B. Define and Train LSTM Model ---
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
    verbose=1 # Changed to verbose=1 to show training progress
)

# --- C. Test on the final, unseen 1-year data ---
y_test_pred_lstm = lstm_model.predict(X_lstm_test_seq_aligned).flatten()
rmse_test = np.sqrt(mean_squared_error(y_test_aligned, y_test_pred_lstm))
# R2 calculation requires a Keras method, but RMSE is the primary metric

print(f"\nModel Evaluation (on Aligned Test Set of {min_len} rows):")
print(f"Final Test Set RMSE (Error in ₹): {rmse_test:.2f}")


# --- 5. Visualization of Final Test Results ---

print("\n--- 4. Visualization of Final Test Results ---")

# Create a DataFrame for easy plotting
test_results_df = pd.DataFrame({
    'Actual': y_test_aligned.values,
    'Predicted': y_test_pred_lstm
})

plt.figure(figsize=(15, 7))
plt.plot(test_results_df['Actual'], label='Actual Close Price', color='blue', linewidth=2)
plt.plot(test_results_df['Predicted'], label='LSTM Prediction', color='red', linestyle='--', linewidth=1.5)

plt.title('Nifty 50 Close Price: Actual vs. LSTM Prediction (Test Set)')
plt.xlabel('Time (Chronological Index)')
plt.ylabel('Nifty 50 Close Price (₹)')
plt.legend()
plt.grid(True)
plt.savefig('Nifty_LSTM_Prediction.png')
# plt.show()