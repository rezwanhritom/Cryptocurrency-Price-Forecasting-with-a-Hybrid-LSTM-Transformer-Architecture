print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Prepare data for visualization
models = list(all_models.keys())
mape_1day = [metrics_1day[m]['MAPE'] for m in models]
mape_7day = [metrics_7day[m]['MAPE'] for m in models]
mape_30day = [metrics_30day[m]['MAPE'] for m in models]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('MAPE Comparison Across All 6 Models - All Prediction Horizons', 
             fontsize=14, fontweight='bold')

# 1-Day
ax1 = axes[0]
colors_1 = ['#FF6B6B' if m != best_1day else '#2ECC71' for m in models]
bars1 = ax1.bar(models, mape_1day, color=colors_1, edgecolor='black', linewidth=1.5)
ax1.set_title('1-Day Ahead Prediction', fontweight='bold', fontsize=12)
ax1.set_ylabel('MAPE (%)', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(mape_1day) * 1.15)
for i, (bar, val) in enumerate(zip(bars1, mape_1day)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 7-Day
ax2 = axes[1]
colors_7 = ['#FF6B6B' if m != best_7day else '#2ECC71' for m in models]
bars7 = ax2.bar(models, mape_7day, color=colors_7, edgecolor='black', linewidth=1.5)
ax2.set_title('7-Day Ahead Prediction', fontweight='bold', fontsize=12)
ax2.set_ylabel('MAPE (%)', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(mape_7day) * 1.15)
for i, (bar, val) in enumerate(zip(bars7, mape_7day)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# 30-Day
ax3 = axes[2]
colors_30 = ['#FF6B6B' if m != best_30day else '#2ECC71' for m in models]
bars30 = ax3.bar(models, mape_30day, color=colors_30, edgecolor='black', linewidth=1.5)
ax3.set_title('30-Day Ahead Prediction', fontweight='bold', fontsize=12)
ax3.set_ylabel('MAPE (%)', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(mape_30day) * 1.15)
for i, (bar, val) in enumerate(zip(bars30, mape_30day)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('02_mape_comparison_all_models.png', dpi=300, bbox_inches='tight')
print("Saved: 02_mape_comparison_all_models.png")
plt.show()