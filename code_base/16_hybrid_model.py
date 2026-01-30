print("="*60)
print("MODEL 5: HYBRID LSTM-TRANSFORMER (NOVEL)")
print("="*60)

class HybridLSTMTransformer(tf.keras.Model):
    """
    Hybrid model combining LSTM (memory) + Transformer (attention)
    with learned fusion gating mechanism
    """
    
    def __init__(self, input_shape, d_model=64):
        super(HybridLSTMTransformer, self).__init__()
        
        # LSTM Branch - captures temporal dependencies
        self.lstm1 = LSTM(64, return_sequences=True)
        self.lstm_dropout = Dropout(0.2)
        self.lstm2 = LSTM(32, return_sequences=False)
        
        # Transformer Branch - captures attention patterns
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        self.input_proj = Dense(d_model)
        self.attention1 = MultiHeadAttention(num_heads=4, key_dim=d_model//4)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.ff1 = Dense(256, activation='relu')
        self.ff1_out = Dense(d_model)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        self.attention2 = MultiHeadAttention(num_heads=4, key_dim=d_model//4)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.ff2 = Dense(256, activation='relu')
        self.ff2_out = Dense(d_model)
        self.norm4 = LayerNormalization(epsilon=1e-6)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        
        # Fusion Gate - learned weighting mechanism
        self.lstm_to_64 = Dense(64)  # FIX: Define this in __init__
        self.fusion_concat = Dense(32, activation='relu')
        self.fusion_gate = Dense(1, activation='sigmoid')  # α ∈ [0, 1]
        self.fusion_project = Dense(64)
        
        # Output layers
        self.output_dropout = Dropout(0.2)
        self.dense32 = Dense(32, activation='relu')
        self.output_dense = Dense(3)
    
    def call(self, x, training=False):
        # LSTM Path
        lstm_seq = self.lstm1(x)
        lstm_seq = self.lstm_dropout(lstm_seq, training=training)
        lstm_out = self.lstm2(lstm_seq)  # (batch, 32)
        
        # Transformer Path
        x_proj = self.input_proj(x)
        x_enc = self.pos_encoding(x_proj)  # (batch, 30, 64)
        
        # First attention block
        attn1 = self.attention1(x_enc, x_enc)
        x_enc = self.norm1(x_enc + attn1)
        ff1 = self.ff1(x_enc)
        ff1 = self.ff1_out(ff1)
        x_enc = self.norm2(x_enc + ff1)
        
        # Second attention block
        attn2 = self.attention2(x_enc, x_enc)
        x_enc = self.norm3(x_enc + attn2)
        ff2 = self.ff2(x_enc)
        ff2 = self.ff2_out(ff2)
        x_enc = self.norm4(x_enc + ff2)
        
        transformer_out = self.pool(x_enc)  # (batch, 64)
        
        # Learned Fusion Gate
        lstm_proj = self.lstm_to_64(lstm_out)  # (batch, 64) - FIX: Use pre-defined layer
        combined = tf.concat([lstm_proj, transformer_out], axis=-1)  # (batch, 128)
        
        gate_input = self.fusion_concat(combined)  # (batch, 32)
        alpha = self.fusion_gate(gate_input)  # (batch, 1) ∈ [0, 1]
        
        # Weighted blend: α × LSTM_proj + (1-α) × Transformer
        fused = alpha * lstm_proj + (1 - alpha) * transformer_out
        fused = self.fusion_project(fused)  # (batch, 64)
        
        # Prediction head
        x = self.output_dropout(fused, training=training)
        x = self.dense32(x)
        x = self.output_dense(x)
        
        return x

# Build model
hybrid_model = HybridLSTMTransformer(input_shape)
hybrid_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nHybrid LSTM-Transformer Model Summary:")
hybrid_model.build(input_shape=(None, *input_shape))
hybrid_model.summary()

# Train
print("\n" + "="*60)
print("TRAINING HYBRID LSTM-TRANSFORMER MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

hybrid_history = hybrid_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nHybrid training complete. Best epoch: {len(hybrid_history.history['loss'])}")
print(f"  Final train loss: {hybrid_history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {hybrid_history.history['val_loss'][-1]:.6f}")

# Save model
hybrid_model.save('hybrid_model.h5')
print("Hybrid model saved as 'hybrid_model.h5'")
