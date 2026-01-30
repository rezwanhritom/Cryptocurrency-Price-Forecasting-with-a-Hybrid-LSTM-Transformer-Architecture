print("\n" + "="*60)
print("STEP 1 COMPLETE: ALL 5 MODELS TRAINED")
print("="*60)

all_models = {
    'LSTM': lstm_model,
    'BiLSTM': bilstm_model,
    'GRU': gru_model,
    'Transformer': transformer_model,
    'Hybrid': hybrid_model
}

all_histories = {
    'LSTM': lstm_history,
    'BiLSTM': bilstm_history,
    'GRU': gru_history,
    'Transformer': transformer_history,
    'Hybrid': hybrid_history
}

print("\n" + "-"*60)
print("TRAINING SUMMARY")
print("-"*60)

summary_data = []
for name, history in all_histories.items():
    hist = history.history
    best_epoch = len(hist['loss'])
    final_train_loss = hist['loss'][-1]
    final_val_loss = hist['val_loss'][-1]
    overfit_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0
    
    summary_data.append({
        'Model': name,
        'Best Epoch': best_epoch,
        'Train Loss': f"{final_train_loss:.6f}",
        'Val Loss': f"{final_val_loss:.6f}",
        'Overfit Ratio': f"{overfit_ratio:.2f}x"
    })
    
    print(f"\n{name}:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Final Train Loss: {final_train_loss:.6f}")
    print(f"  Final Val Loss: {final_val_loss:.6f}")
    print(f"  Overfit Ratio: {overfit_ratio:.2f}x")

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*60)
print("TRAINING SUMMARY TABLE")
print("="*60)
print(summary_df.to_string(index=False))

print("\nAll 5 models trained and saved")