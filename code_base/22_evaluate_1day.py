print("\n" + "="*60)
print("EVALUATION: 1-DAY PREDICTION HORIZON")
print("="*60)

# Extract 1-day predictions and targets
y_test_1day = y_test[:, 0]  # First column is 1-day ahead
metrics_1day = {}

print("\nCalculating metrics for 1-day ahead predictions...\n")

for model_name, y_pred in all_predictions.items():
    y_pred_1day = y_pred[:, 0]
    
    mape = calculate_mape(y_test_1day, y_pred_1day)
    rmse = calculate_rmse(y_test_1day, y_pred_1day)
    mae = calculate_mae(y_test_1day, y_pred_1day)
    dir_acc = calculate_directional_accuracy(y_test_1day, y_pred_1day)
    r2 = calculate_r2(y_test_1day, y_pred_1day)
    
    metrics_1day[model_name] = {
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
df_1day = pd.DataFrame(metrics_1day).T
df_1day = df_1day.round(4)
print("="*60)
print("1-DAY METRICS TABLE")
print("="*60)
print(df_1day)
print("\nBest model (1-day MAPE): ", df_1day['MAPE'].idxmin())