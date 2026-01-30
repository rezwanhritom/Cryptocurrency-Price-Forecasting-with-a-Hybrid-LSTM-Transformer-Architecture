print("\n" + "="*60)
print("DEFINING EVALUATION METRICS")
print("="*60)

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def calculate_directional_accuracy(y_true, y_pred):
    """Direction accuracy: % of correct up/down predictions"""
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    accuracy = np.mean(true_direction == pred_direction) * 100
    return accuracy

def calculate_r2(y_true, y_pred):
    """R-squared score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

print("\n✓ Metrics defined:")
print("  • MAPE (Mean Absolute Percentage Error)")
print("  • RMSE (Root Mean Squared Error)")
print("  • MAE (Mean Absolute Error)")
print("  • Directional Accuracy")
print("  • R² Score")