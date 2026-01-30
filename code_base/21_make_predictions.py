print("\n" + "="*60)
print("MAKING PREDICTIONS ON TEST SET")
print("="*60)

# Store predictions for all models
all_predictions = {}

print(f"\nTest set shape: {X_test.shape}")
print(f"Test targets shape: {y_test.shape}")

for model_name, model in all_models.items():
    print(f"\nPredicting with {model_name}...")
    y_pred = model.predict(X_test, verbose=0)
    all_predictions[model_name] = y_pred
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}, Mean: {y_pred.mean():.2f}")

print("\nAll predictions complete!")