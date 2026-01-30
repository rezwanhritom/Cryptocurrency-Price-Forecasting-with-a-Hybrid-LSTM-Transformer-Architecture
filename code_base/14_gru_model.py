print("="*60)
print("MODEL 3: GRU (Gated Recurrent Unit)")
print("="*60)

from tensorflow.keras.layers import GRU

def build_gru_model(input_shape):
    """Build GRU model"""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(3)
    ])
    return model

# Build model
gru_model = build_gru_model(input_shape)
gru_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nGRU Model Summary:")
gru_model.summary()

# Train
print("\n" + "="*60)
print("TRAINING GRU MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

gru_history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nGRU training complete. Best epoch: {len(gru_history.history['loss'])}")
print(f"  Final train loss: {gru_history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {gru_history.history['val_loss'][-1]:.6f}")

# Save model
gru_model.save('gru_model.h5')
print("GRU model saved as 'gru_model.h5'")
