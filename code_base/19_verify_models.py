print("\nAll 5 models already in memory from training!")
print("\nModels ready for evaluation:")

all_models = {
    'LSTM': lstm_model,
    'BiLSTM': bilstm_model,
    'GRU': gru_model,
    'Transformer': transformer_model,
    'Hybrid': hybrid_model
}

for name in all_models.keys():
    print(f"  • {name}")

print("\n" + "="*60)
print("MODELS READY FOR EVALUATION")
print("="*60)

print("\nVerifying models are ready...")
for name, model in all_models.items():
    print(f"  {name}: {model is not None}")