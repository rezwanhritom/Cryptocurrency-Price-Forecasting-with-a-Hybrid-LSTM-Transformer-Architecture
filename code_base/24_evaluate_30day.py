print("\n" + "="*60)
print("EVALUATION: 30-DAY PREDICTION HORIZON")
print("="*60)

# Extract 30-day predictions and targets
y_test_30day = y_test[:, 2]  # Third column is 30-day ahead
metrics_30day = {}

print("\nCalculating metrics for 30-day ahead predictions...\n")

for model_name, y_pred in all_predictions.items():
    y_pred_30day = y_pred[:, 2]
    
    mape = calculate_mape(y_test_30day, y_pred_30day)
    rmse = calculate_rmse(y_test_30day, y_pred_30day)
    mae = calculate_mae(y_test_30day, y_pred_30day)
    dir_acc = calculate_directional_accuracy(y_test_30day, y_pred_30day)
    r2 = calculate_r2(y_test_30day, y_pred_30day)
    
    metrics_30day[model_name] = {
        'MAPE': mape,
        'RMSE': rmse,
        'MAE': mae,
        'Dir_Acc': dir_acc,
        'R2': r2
    }
    
    print(f"{model_name}:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Directional Accuracy: {dir_acc:.2f}%")
    print(f"  R²: {r2:.4f}\n")

# Create comparison table
df_30day = pd.DataFrame(metrics_30day).T
df_30day = df_30day.round(4)
print("="*60)
print("30-DAY METRICS TABLE")
print("="*60)
print(df_30day)
print("\nBest model (30-day MAPE): ", df_30day['MAPE'].idxmin())