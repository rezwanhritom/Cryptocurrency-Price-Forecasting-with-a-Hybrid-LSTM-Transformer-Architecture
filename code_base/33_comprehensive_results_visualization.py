print("\n" + "="*80)
print("CREATING COMPREHENSIVE RESULTS VISUALIZATION")
print("="*80)

# Create 3x2 grid of all key comparisons
fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('Hybrid LSTM-Transformer: Comprehensive Results Summary', 
             fontsize=18, fontweight='bold', y=0.995)

# ============================================================================
# Plot 1: MAPE Comparison (1-Day)
# ============================================================================
ax1 = axes[0, 0]
models_list = list(original_results.keys())
mape_1day_values = [original_results[m]['MAPE_1day'] for m in models_list]
colors = ['#E74C3C' if m != 'Hybrid' else '#2ECC71' for m in models_list]

bars1 = ax1.bar(models_list, mape_1day_values, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('MAPE (%)', fontweight='bold', fontsize=11)
ax1.set_title('1-Day Prediction MAPE', fontweight='bold', fontsize=12)
ax1.set_ylim([70, 90])
ax1.grid(True, alpha=0.3, axis='y')

for bar, mape in zip(bars1, mape_1day_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mape:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 2: R² Score Comparison
# ============================================================================
ax2 = axes[0, 1]
r2_values = [original_results[m]['R2'] for m in models_list]
colors2 = ['#E74C3C' if m != 'Hybrid' else '#2ECC71' for m in models_list]

bars2 = ax2.bar(models_list, r2_values, color=colors2, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('R² Score', fontweight='bold', fontsize=11)
ax2.set_title('Model Fit Quality (R²)', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

for bar, r2 in zip(bars2, r2_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{r2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 3: All MAPE Horizons
# ============================================================================
ax3 = axes[1, 0]
x_pos = np.arange(len(models_list))
width = 0.25

mape_1d = [original_results[m]['MAPE_1day'] for m in models_list]
mape_7d = [original_results[m]['MAPE_7day'] for m in models_list]
mape_30d = [original_results[m]['MAPE_30day'] for m in models_list]

ax3.bar(x_pos - width, mape_1d, width, label='1-Day', color='#3498DB', edgecolor='black')
ax3.bar(x_pos, mape_7d, width, label='7-Day', color='#F39C12', edgecolor='black')
ax3.bar(x_pos + width, mape_30d, width, label='30-Day', color='#E74C3C', edgecolor='black')

ax3.set_ylabel('MAPE (%)', fontweight='bold', fontsize=11)
ax3.set_title('MAPE Across All Prediction Horizons', fontweight='bold', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models_list)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Directional Accuracy
# ============================================================================
ax4 = axes[1, 1]
dir_acc = [original_results[m]['Directional_Accuracy'] for m in models_list]
colors4 = ['#E74C3C' if m != 'Hybrid' else '#2ECC71' for m in models_list]

bars4 = ax4.bar(models_list, dir_acc, color=colors4, edgecolor='black', linewidth=1.5)
ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)', alpha=0.7)
ax4.set_ylabel('Directional Accuracy (%)', fontweight='bold', fontsize=11)
ax4.set_title('Direction Prediction Accuracy', fontweight='bold', fontsize=12)
ax4.set_ylim([48, 58])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars4, dir_acc):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============================================================================
# Plot 5: Hybrid vs LSTM Improvement Breakdown
# ============================================================================
ax5 = axes[2, 0]

metrics_names = ['MAPE\n1-Day', 'MAPE\n7-Day', 'MAPE\n30-Day', 'R²\n(×100)', 'Dir. Acc.\n(×1%)']
lstm_vals = [
    original_results['LSTM']['MAPE_1day'],
    original_results['LSTM']['MAPE_7day'],
    original_results['LSTM']['MAPE_30day'],
    original_results['LSTM']['R2'] * 100,
    original_results['LSTM']['Directional_Accuracy']
]
hybrid_vals = [
    original_results['Hybrid']['MAPE_1day'],
    original_results['Hybrid']['MAPE_7day'],
    original_results['Hybrid']['MAPE_30day'],
    original_results['Hybrid']['R2'] * 100,
    original_results['Hybrid']['Directional_Accuracy']
]

x_pos2 = np.arange(len(metrics_names))
width2 = 0.35

ax5.bar(x_pos2 - width2/2, lstm_vals, width2, label='LSTM', color='#E74C3C', edgecolor='black')
ax5.bar(x_pos2 + width2/2, hybrid_vals, width2, label='Hybrid', color='#2ECC71', edgecolor='black')

ax5.set_ylabel('Score', fontweight='bold', fontsize=11)
ax5.set_title('Hybrid vs LSTM: Metric-by-Metric Comparison', fontweight='bold', fontsize=12)
ax5.set_xticks(x_pos2)
ax5.set_xticklabels(metrics_names, fontsize=10)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 6: Overall Performance Ranking
# ============================================================================
ax6 = axes[2, 1]
ax6.axis('off')

# Create ranking text
ranking_text = "MODEL RANKING (by 1-Day MAPE)\n\n"
sorted_models = sorted([(m, original_results[m]['MAPE_1day']) for m in models_list], 
                       key=lambda x: x[1])

medals = ['1', '2', '3', '4', '5']
for rank, (model, mape) in enumerate(sorted_models):
    ranking_text += f"{medals[rank]} {rank+1}. {model:20s} {mape:6.2f}%\n"

ranking_text += "\n" + "="*50 + "\n"
ranking_text += f"KEY RESULT: Hybrid outperforms LSTM baseline\n"
ranking_text += f"  • Improvement: {((original_results['LSTM']['MAPE_1day'] - original_results['Hybrid']['MAPE_1day']) / original_results['LSTM']['MAPE_1day'] * 100):.2f}%\n"
ranking_text += f"  • Better R²: {original_results['Hybrid']['R2'] - original_results['LSTM']['R2']:.4f}\n"
ranking_text += f"  • Novel Architecture: LSTM + Transformer + Learned Fusion\n"

ax6.text(0.05, 0.95, ranking_text, transform=ax6.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('10_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("Saved: 10_comprehensive_results.png")
plt.show()

print("\nComprehensive visualization complete")
