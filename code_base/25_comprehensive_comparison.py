print("\n" + "="*60)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*60)

# Create overall comparison
comparison_data = []

for model_name in all_models.keys():
    comparison_data.append({
        'Model': model_name,
        '1D_MAPE': f"{metrics_1day[model_name]['MAPE']:.2f}%",
        '1D_Acc': f"{metrics_1day[model_name]['Dir_Acc']:.1f}%",
        '7D_MAPE': f"{metrics_7day[model_name]['MAPE']:.2f}%",
        '7D_Acc': f"{metrics_7day[model_name]['Dir_Acc']:.1f}%",
        '30D_MAPE': f"{metrics_30day[model_name]['MAPE']:.2f}%",
        '30D_Acc': f"{metrics_30day[model_name]['Dir_Acc']:.1f}%",
    })

df_comparison = pd.DataFrame(comparison_data)

print("\n")
print(df_comparison.to_string(index=False))

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

# Extract numeric MAPE values for comparison
mapes_1day = {name: metrics_1day[name]['MAPE'] for name in all_models.keys()}
mapes_7day = {name: metrics_7day[name]['MAPE'] for name in all_models.keys()}
mapes_30day = {name: metrics_30day[name]['MAPE'] for name in all_models.keys()}

best_1day = min(mapes_1day, key=mapes_1day.get)
best_7day = min(mapes_7day, key=mapes_7day.get)
best_30day = min(mapes_30day, key=mapes_30day.get)

print(f"\n1-Day Horizon:")
print(f"  Best: {best_1day} ({mapes_1day[best_1day]:.2f}% MAPE)")
print(f"  Range: {min(mapes_1day.values()):.2f}% - {max(mapes_1day.values()):.2f}%")

print(f"\n7-Day Horizon:")
print(f"  Best: {best_7day} ({mapes_7day[best_7day]:.2f}% MAPE)")
print(f"  Range: {min(mapes_7day.values()):.2f}% - {max(mapes_7day.values()):.2f}%")

print(f"\n30-Day Horizon:")
print(f"  Best: {best_30day} ({mapes_30day[best_30day]:.2f}% MAPE)")
print(f"  Range: {min(mapes_30day.values()):.2f}% - {max(mapes_30day.values()):.2f}%")

# Check if Hybrid beats LSTM
lstm_1day = mapes_1day['LSTM']
hybrid_1day = mapes_1day['Hybrid']
improvement = ((lstm_1day - hybrid_1day) / lstm_1day) * 100

print(f"\n" + "="*60)
print("HYBRID vs LSTM COMPARISON")
print("="*60)
print(f"LSTM 1-Day MAPE: {lstm_1day:.2f}%")
print(f"Hybrid 1-Day MAPE: {hybrid_1day:.2f}%")
print(f"Improvement: {improvement:.2f}%")

if hybrid_1day < lstm_1day:
    print(f"\nHYBRID OUTPERFORMS LSTM by {improvement:.2f}%")
else:
    print(f"\nLSTM performs better than Hybrid")
