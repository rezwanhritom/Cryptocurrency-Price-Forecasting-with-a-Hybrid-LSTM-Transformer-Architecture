print("\nCreating R² and RMSE heatmap...")

# Prepare R² data
r2_data = []
for model_name in models:
    r2_data.append([
        metrics_1day[model_name]['R2'],
        metrics_7day[model_name]['R2'],
        metrics_30day[model_name]['R2']
    ])

r2_array = np.array(r2_data)

# Create heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance: R² Score & RMSE Heatmaps', fontsize=14, fontweight='bold')

# R² Heatmap
ax1 = axes[0]
im1 = ax1.imshow(r2_array, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1)
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(['1-Day', '7-Day', '30-Day'])
ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models)
ax1.set_title('R² Score (Higher is Better)', fontweight='bold')

# Add values
for i in range(len(models)):
    for j in range(3):
        text = ax1.text(j, i, f'{r2_array[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im1, ax=ax1, label='R² Score')

# RMSE Heatmap
rmse_data = []
for model_name in models:
    rmse_data.append([
        metrics_1day[model_name]['RMSE'],
        metrics_7day[model_name]['RMSE'],
        metrics_30day[model_name]['RMSE']
    ])

rmse_array = np.array(rmse_data)

ax2 = axes[1]
im2 = ax2.imshow(rmse_array, cmap='YlOrRd_r', aspect='auto')
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['1-Day', '7-Day', '30-Day'])
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models)
ax2.set_title('RMSE (Lower is Better)', fontweight='bold')

# Add values
for i in range(len(models)):
    for j in range(3):
        text = ax2.text(j, i, f'{rmse_array[i, j]:.0f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im2, ax=ax2, label='RMSE')

plt.tight_layout()
plt.savefig('04_r2_rmse_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: 04_r2_rmse_heatmap.png")
plt.show()