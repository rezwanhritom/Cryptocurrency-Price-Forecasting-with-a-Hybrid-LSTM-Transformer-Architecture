print("="*60)
print("MODEL 4: TRANSFORMER (Attention-Based)")
print("="*60)

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model=64):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        
        # FIX: Convert to float32 BEFORE division
        dim_indices = tf.range(0, self.d_model, 2, dtype=tf.float32)
        
        # Positional encoding formula
        angle_rates = 1.0 / tf.pow(10000.0, dim_indices / tf.cast(self.d_model, tf.float32))
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices
        sines = tf.sin(angle_rads)
        # Apply cos to odd indices  
        cosines = tf.cos(angle_rads)
        
        # Interleave sin and cos
        pos_encoding = tf.reshape(
            tf.stack([sines, cosines], axis=2),
            [seq_len, -1]
        )
        
        # Handle odd d_model (take first d_model dimensions)
        pos_encoding = pos_encoding[:, :self.d_model]
        
        # Expand batch dimension and add to input
        return x + tf.expand_dims(pos_encoding, 0)

def build_transformer_model(input_shape, d_model=64, num_heads=4, num_layers=2):
    """Build Transformer model"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Project input to d_model dimension
    x = Dense(d_model)(inputs)
    
    # Positional encoding
    x = PositionalEncoding(d_model=d_model)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model//num_heads,
            dropout=0.1
        )(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ff_output = Dense(256, activation='relu')(x)
        ff_output = Dense(d_model)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output layers
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(3)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Build model
transformer_model = build_transformer_model(input_shape)
transformer_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nTransformer Model Summary:")
transformer_model.summary()

# Train
print("\n" + "="*60)
print("TRAINING TRANSFORMER MODEL")
print("="*60)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

transformer_history = transformer_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nTransformer training complete. Best epoch: {len(transformer_history.history['loss'])}")
print(f"  Final train loss: {transformer_history.history['loss'][-1]:.6f}")
print(f"  Final val loss: {transformer_history.history['val_loss'][-1]:.6f}")

# Save model
transformer_model.save('transformer_model.h5')
print("Transformer model saved as 'transformer_model.h5'")
