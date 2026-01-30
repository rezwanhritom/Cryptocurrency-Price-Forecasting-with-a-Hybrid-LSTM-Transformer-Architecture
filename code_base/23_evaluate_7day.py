print("\n" + "="*60)
print("EVALUATION: 7-DAY PREDICTION HORIZON")
print("="*60)

# Extract 7-day predictions and targets
y_test_7day = y_test[:, 1]  # Second column is 7-day ahead
metrics_7day = {}

print("\nCalculating metrics for 7-day ahead predictions...\n")

for model_name, y_pred in all_predictions.items():
    y_pred_7day = y_pred[:, 1]
    
    mape = calculate_mape(y_test_7day, y_pred_7day)
    rmse = calculate_rmse(y_test_7day, y_pred_7day)
    mae = calculate_mae(y_test_7day, y_pred_7day)
    dir_acc = calculate_directional_accuracy(y_test_7day, y_pred_7day)
    r2 = calculate_r2(y_test_7day, y_pred_7day)
    
    metrics_7day[model_name] = {
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
df_7day = pd.DataFrame(metrics_7day).T
df_7day = df_7day.round(4)
print("="*60)
print("7-DAY METRICS TABLE")
print("="*60)
print(df_7day)
print("\nBest model (7-day MAPE): ", df_7day['MAPE'].idxmin())