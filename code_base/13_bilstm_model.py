print("="*60)
print("MODEL 2: BILSTM (Bidirectional LSTM)")
print("="*60)

from tensorflow.keras.layers import Bidirectional

def build_bilstm_model(input_shape):
    """Build BiLSTM model"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(3)
    ])
    return model

# Build model
bilstm_model = build_bilstm_model(input_shape)
bilstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nBiLSTM Model Summary:")
bilstm_model.summary()

# Train
print("\n" + "="*60)
print("TRAINING BILSTM MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

bilstm_history = bilstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nBiLSTM training complete. Best epoch: {len(bilstm_history.history['loss'])}")
print(f"  Final train loss: {bilstm_history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {bilstm_history.history['val_loss'][-1]:.6f}")

# Save model
bilstm_model.save('bilstm_model.h5')
print("BiLSTM model saved as 'bilstm_model.h5'")
