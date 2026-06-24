import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    tf.get_logger().setLevel('ERROR')
except:
    pass

# Set seeds for reproducibility
np.random.seed(42)

# --- Configuration (Update File Paths) ---
FILE_PATHS = [
 'NIFTY 50-11-10-2020-to-11-10-2021.csv',
 'NIFTY 50-11-10-2021-to-11-10-2022.csv',
 'NIFTY 50-11-10-2022-to-11-10-2023.csv',
 'NIFTY 50-11-10-2023-to-11-10-2024.csv',
 'NIFTY 50-11-10-2024-to-11-10-2025.csv'
]
TARGET_COLUMN = 'Close'
N_JOBS = -1

# Mock/Placeholder functions (kept for full code completeness)
def create_and_process_features(df): 
    raise NotImplementedError("Requires full data loading logic.")
def load_and_prepare_data(*args, **kwargs): raise FileNotFoundError("Data files not found.")
def split_data(*args, **kwargs): pass
def run_model(*args, **kwargs): pass
def create_lstm_sequences(*args, **kwargs): pass
def build_lstm_model(input_shape): pass
def calculate_directional_accuracy(y_actual, y_pred, y_prev_close): pass


# Function to attach a label above each bar
def autolabel(ax, rects, format_str, log_scale=False):
    """
    Attaches labels above bars, adjusting vertical positioning for log scale.
    """
    for rect in rects:
        height = rect.get_height()
        
        # Adjust vertical position for annotation based on scale
        if log_scale:
            # For log scale, annotation needs more vertical distance in the log space
            y_pos = height * 1.05 
        else:
            y_pos = height
            
        # Determine annotation offset based on axis limits (standard approach)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01 
        
        ax.annotate(format_str.format(height),
              xy=(rect.get_x() + rect.get_width() / 2, y_pos),
              xytext=(0, offset * 5),
              textcoords="offset points",
              ha='center', va='bottom', fontsize=9)

# --- NEW PLOTTING FUNCTIONS (Separated Plots) ---

def plot_rmse(model_names, train_rmses, test_rmses, abbreviated_names):
    """Generates and saves the RMSE comparison plot with a LOGARITHMIC Y-axis."""
    x = np.arange(len(model_names))
    width = 0.35
    colors_rmse = ['skyblue', 'salmon']
    
    fig, ax = plt.subplots(figsize=(12, 7)) 
    
    rects1_rmse = ax.bar(x - width/2, train_rmses, width, label='Training RMSE', color=colors_rmse[0], edgecolor='black')
    rects2_rmse = ax.bar(x + width/2, test_rmses, width, label='Test RMSE', color=colors_rmse[1], edgecolor='black')

    # *** FIX: Apply Logarithmic Scale to Y-axis ***
    ax.set_yscale('log')
    # Set realistic limits (start > 0 since log(0) is undefined)
    # The minimum value is 5, so start at 1 or 2. The maximum is ~1900, so end at 2000-3000.
    ax.set_ylim(1, 3000) 
    # Add a custom tick label to clearly show the range
    ax.set_yticks([1, 10, 100, 1000, 3000])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter()) # Keep labels as numbers, not scientific notation

    ax.set_ylabel('RMSE (Root Mean Square Error in ₹ - LOG SCALE)', fontsize=12)
    ax.set_xlabel('Regression Model', fontsize=12)
    ax.set_title('Training vs. Test RMSE Comparison (Lower is Better - Logarithmic Scale)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(abbreviated_names, rotation=0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.5)
    
    # Pass log_scale=True to autolabel for position adjustment
    autolabel(ax, rects1_rmse, '{:.2f}', log_scale=True)
    autolabel(ax, rects2_rmse, '{:.2f}', log_scale=True)

    plt.tight_layout()
    plt.savefig('rmse_comparison_log.png')
    plt.close(fig) 
    print("Saved RMSE comparison plot with log scale as rmse_comparison_log.png")

def plot_accuracy(model_names, train_accs, test_accs, abbreviated_names):
    """Generates and saves the Directional Accuracy comparison plot."""
    x = np.arange(len(model_names))
    width = 0.35
    colors_acc = ['lightgreen', 'darkgreen']
    
    fig, ax = plt.subplots(figsize=(12, 7)) 
    
    rects1_acc = ax.bar(x - width/2, train_accs, width, label='Training Directional Accuracy', color=colors_acc[0], edgecolor='black')
    rects2_acc = ax.bar(x + width/2, test_accs, width, label='Test Directional Accuracy', color=colors_acc[1], edgecolor='black')

    ax.set_ylabel('Directional Accuracy (Proportion)', fontsize=12)
    ax.set_xlabel('Regression Model', fontsize=12)
    ax.set_title('Training vs. Test Directional Accuracy (Higher is Better)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(abbreviated_names, rotation=0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.5)
    ax.set_ylim(bottom=0.45, top=1.0) 
    
    autolabel(ax, rects1_acc, '{:.3f}')
    autolabel(ax, rects2_acc, '{:.3f}')

    plt.tight_layout()
    plt.savefig('directional_accuracy.png')
    plt.close(fig) 
    print("Saved directional accuracy plot as directional_accuracy.png")

# --- MAIN EXECUTION ---

def main():
    
    results = []
    
    try:
        # Load data (Skipped in this environment)
        X, y, y_scaler, prev_close = load_and_prepare_data(FILE_PATHS, TARGET_COLUMN)
        
    except FileNotFoundError:
        # Mock data transcribed from the original image (Max Test RMSE ~1900)
        print("\n--- WARNING: File not found. Using mock data for log scale demonstration. ---")
        
        results = [
            {'model_name': 'Linear Regression', 'train_rmse': 78.00, 'test_rmse': 115.00, 'train_acc': 0.883, 'test_acc': 0.886},
            {'model_name': 'Random Forest', 'train_rmse': 5.00, 'test_rmse': 1850.00, 'train_acc': 0.856, 'test_acc': 0.516},
            {'model_name': 'XGBoost', 'train_rmse': 10.00, 'test_rmse': 1900.00, 'train_acc': 0.953, 'test_acc': 0.514},
            {'model_name': 'LSTM', 'train_rmse': 250.00, 'test_rmse': 320.00, 'train_acc': 0.583, 'test_acc': 0.465},
            {'model_name': 'RF + XGBoost Hybrid', 'train_rmse': 70.00, 'test_rmse': 1870.00, 'train_acc': 0.918, 'test_acc': 0.514},
            {'model_name': 'LR + LSTM Blended Hybrid', 'train_rmse': 120.00, 'test_rmse': 250.00, 'train_acc': 0.771, 'test_acc': 0.576}
        ]


    model_names = [res['model_name'] for res in results]
    train_rmses = [res['train_rmse'] for res in results]
    test_rmses = [res['test_rmse'] for res in results]
    train_accs = [res['train_acc'] for res in results]
    test_accs = [res['test_acc'] for res in results]
    
    abbreviated_names = [
      'LR', 'RF', 'XGB', 'LSTM', 'RF+XGB Hybrid', 'LR+LSTM Hybrid'
    ]
    
    print("\n--- Generating Separate Comparison Charts (RMSE and Accuracy) ---")
    
    # Plot 1: RMSE (Now with Log Scale)
    plot_rmse(model_names, train_rmses, test_rmses, abbreviated_names)
    
    # Plot 2: Directional Accuracy (No change needed)
    plot_accuracy(model_names, train_accs, test_accs, abbreviated_names)

if __name__ == '__main__':
  main()