print("\nCreating prediction vs actual comparison for Hybrid model...")

y_pred_hybrid = all_predictions['Hybrid']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Hybrid Model: Predicted vs Actual (Test Set)', fontsize=14, fontweight='bold')

horizons = ['1-Day', '7-Day', '30-Day']
test_targets = [y_test[:, 0], y_test[:, 1], y_test[:, 2]]
predictions = [y_pred_hybrid[:, 0], y_pred_hybrid[:, 1], y_pred_hybrid[:, 2]]

for idx, (ax, horizon, actual, pred) in enumerate(zip(axes, horizons, test_targets, predictions)):
    # Scatter plot
    ax.scatter(actual, pred, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Metrics
    mape = calculate_mape(actual, pred)
    r2 = calculate_r2(actual, pred)
    
    ax.set_xlabel('Actual Price', fontweight='bold')
    ax.set_ylabel('Predicted Price', fontweight='bold')
    ax.set_title(f'{horizon}\nMAPE: {mape:.2f}% | R²: {r2:.4f}', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('05_hybrid_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Saved: 05_hybrid_predictions_vs_actual.png")
plt.show()