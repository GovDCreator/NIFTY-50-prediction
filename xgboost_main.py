import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration (Update File Paths) ---
# Assuming your 5 CSV files are named year1.csv, year2.csv, ...
# **CRITICAL:** Update these file names to match your data.
FILE_PATHS = ['NIFTY 50-11-10-2020-to-11-10-2021.csv','NIFTY 50-11-10-2021-to-11-10-2022.csv','NIFTY 50-11-10-2022-to-11-10-2023.csv','NIFTY 50-11-10-2023-to-11-10-2024.csv','NIFTY 50-11-10-2024-to-11-10-2025.csv'] 
TARGET_COLUMN = 'Close'
N_ITER = 30     # Number of parameter settings sampled (your requested 30 models)
CV_SPLITS = 5   # Number of splits for TimeSeriesSplit CV
N_JOBS = -1     # Use all CPU cores

# --- 1. Data Consolidation and Feature Engineering Function ---

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

# --- 2. Data Loading, Feature Creation, and Scaling ---

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
print(df_processed.columns)
y = df_processed[TARGET_COLUMN]
X = df_processed.drop(columns=[TARGET_COLUMN])

# Identify scaling features (all except binary/OHE)
binary_features = [col for col in X.columns if col.startswith('Is_')]
features_to_scale = [col for col in X.columns if col not in binary_features]

# Apply MinMaxScaler on the entire X dataset
scaler = MinMaxScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

print(f"Total rows available after processing: {len(X)}")


# --- 3. Time-Series Train/Validate/Test Split ---

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


# --- 4. Hyperparameter Tuning using RandomizedSearchCV ---

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

# --- 5. Final Training and Testing ---

print("\n--- 4. Final Model Training and Testing ---")

# 1. Evaluate best model on the Validation Pool (0.5 year data)
best_model_val = random_search.best_estimator_
y_val_pred = best_model_val.predict(X_val_pool)
rmse_val = np.sqrt(mean_squared_error(y_val_pool, y_val_pred))
print(f"RMSE on Validation Pool (0.5y data) using best params: {rmse_val:.2f}")

# 2. Final Training: Combine Train Base and Validation Pool
X_train_final = pd.concat([X_train_base, X_val_pool])
y_train_final = pd.concat([y_train_base, y_val_pool])

# Train the final model using the best hyperparameters found
final_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_jobs=N_JOBS,
    **best_params 
)
final_xgb_model.fit(X_train_final, y_train_final)

# 3. Test on the final, unseen 1-year data
y_test_pred = final_xgb_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = final_xgb_model.score(X_test, y_test)

print(f"\nFinal Model Trained on: {len(X_train_final)} rows")
print(f"Final Test Set RMSE (Error in ₹): {rmse_test:.2f}")
print(f"Final Test Set R-squared (Model Fit): {r2_test:.4f}")

# --- 6. Visualization ---

print("\n--- 5. Visualization of Final Test Results ---")

# Create a DataFrame for easy plotting
test_results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred
})

# Add a date index for better x-axis visualization (optional, requires a bit more complexity with the Date column,
# but for simplicity, we'll use the array index for now, as the data is chronological)

plt.figure(figsize=(15, 7))
plt.plot(test_results_df['Actual'], label='Actual Close Price', color='blue', linewidth=2)
plt.plot(test_results_df['Predicted'], label='Predicted Close Price', color='red', linestyle='--', linewidth=1.5)

plt.title('Nifty 50 Close Price: Actual vs. Final Model Prediction (Test Set)')
plt.xlabel('Time (Chronological Index)')
plt.ylabel('Nifty 50 Close Price (₹)')
plt.legend()
plt.grid(True)
plt.show()