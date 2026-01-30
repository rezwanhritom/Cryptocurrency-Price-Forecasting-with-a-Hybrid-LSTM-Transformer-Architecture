print("Creating directional accuracy comparison...")

# Prepare data
dir_acc_1day = [metrics_1day[m]['Dir_Acc'] for m in models]
dir_acc_7day = [metrics_7day[m]['Dir_Acc'] for m in models]
dir_acc_30day = [metrics_30day[m]['Dir_Acc'] for m in models]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Directional Accuracy - All Models & Horizons', 
             fontsize=14, fontweight='bold')

# 1-Day
ax1 = axes[0]
bars1 = ax1.bar(models, dir_acc_1day, color='#3498DB', edgecolor='black', linewidth=1.5)
ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Guess (50%)')
ax1.set_title('1-Day Ahead', fontweight='bold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)
ax1.legend()
for bar, val in zip(bars1, dir_acc_1day):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 7-Day
ax2 = axes[1]
bars7 = ax2.bar(models, dir_acc_7day, color='#9B59B6', edgecolor='black', linewidth=1.5)
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Guess (50%)')
ax2.set_title('7-Day Ahead', fontweight='bold', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)
ax2.legend()
for bar, val in zip(bars7, dir_acc_7day):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# 30-Day
ax3 = axes[2]
bars30 = ax3.bar(models, dir_acc_30day, color='#E67E22', edgecolor='black', linewidth=1.5)
ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Guess (50%)')
ax3.set_title('30-Day Ahead', fontweight='bold', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)
ax3.legend()
for bar, val in zip(bars30, dir_acc_30day):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('03_directional_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 03_directional_accuracy_comparison.png")
plt.show()
