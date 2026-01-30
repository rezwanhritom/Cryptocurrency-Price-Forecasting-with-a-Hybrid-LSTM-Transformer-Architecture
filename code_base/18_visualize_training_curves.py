print("="*60)
print("VISUALIZING TRAINING CURVES")
print("="*60)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Training Curves: All 5 Models', fontsize=16, fontweight='bold')

model_names = ['LSTM', 'BiLSTM', 'GRU', 'Transformer', 'Hybrid']
positions = [(0), (1), (2), (3), (4)]

for model_name, pos in zip(model_names, positions):
    ax = axes[pos]
    hist = all_histories[model_name].history
    
    epochs_range = range(1, len(hist['loss']) + 1)
    
    ax.plot(epochs_range, hist['loss'], 'b-', linewidth=2, label='Train Loss')
    ax.plot(epochs_range, hist['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax.fill_between(epochs_range, hist['loss'], hist['val_loss'], alpha=0.2)
    
    ax.set_title(f'{model_name}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('training_curves_all_5_models.png', dpi=300, bbox_inches='tight')
print("Saved: training_curves_all_5_models.png")
plt.show()

print("\nModels saved:")
print("  • lstm_model.h5")
print("  • bilstm_model.h5")
print("  • gru_model.h5")
print("  • transformer_model.h5")
print("  • hybrid_model.h5")