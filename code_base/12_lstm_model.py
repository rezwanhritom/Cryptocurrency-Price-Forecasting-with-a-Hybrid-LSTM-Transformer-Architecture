print("="*60)
print("MODEL 1: LSTM BASELINE")
print("="*60)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(3)  # 1-day, 7-day, 30-day predictions
    ])
    return model

# Get input shape
input_shape = (X_train.shape[1], X_train.shape[2])  # (30, 15)
print(f"Input shape: {input_shape}")

# Build model
lstm_model = build_lstm_model(input_shape)
lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nLSTM Model Summary:")
lstm_model.summary()

# Train
print("\n" + "="*60)
print("TRAINING LSTM MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nLSTM training complete. Best epoch: {len(lstm_history.history['loss'])}")
print(f"  Final train loss: {lstm_history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {lstm_history.history['val_loss'][-1]:.6f}")

# Save model
lstm_model.save('lstm_model.h5')
print("LSTM model saved as 'lstm_model.h5'")