print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*60)

from scipy import stats

print("\nTesting if Hybrid is statistically significantly better than LSTM...")
print("Using paired t-test on 1-Day predictions\n")

# Get predictions
y_pred_lstm_1day = all_predictions['LSTM'][:, 0]
y_pred_hybrid_1day = all_predictions['Hybrid'][:, 0]

# Calculate errors
lstm_errors = np.abs(y_test_1day - y_pred_lstm_1day)
hybrid_errors = np.abs(y_test_1day - y_pred_hybrid_1day)

# Paired t-test
t_stat, p_value = stats.ttest_rel(lstm_errors, hybrid_errors)

print(f"LSTM MAE: {lstm_errors.mean():.2f}")
print(f"Hybrid MAE: {hybrid_errors.mean():.2f}")
print(f"Difference: {(lstm_errors.mean() - hybrid_errors.mean()):.2f}")
print(f"\nPaired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"\nSIGNIFICANT: Hybrid is statistically significantly better than LSTM (p < 0.05)")
else:
    print(f"\nNOT SIGNIFICANT: No statistical difference (p >= 0.05)")
    print(f"(This is still okay - practical improvement matters in finance)")