print("="*60)
print("CREATING SEQUENCES (30-DAY LOOKBACK)")
print("="*60)

# Dictionary to store sequence data
sequence_data = {}

def create_sequences(normalized_df, ticker_name, lookback=30, pred_horizons=[1, 7, 30]):
    print(f"\nCreating sequences for {ticker_name}...")
    
    if normalized_df is None:
        print(f"{ticker_name} data is None, skipping")
        return None, None
    
    # Get feature columns (exclude Date and Close_Original)
    feature_cols = [col for col in normalized_df.columns if col not in ['Date', 'Close_Original']]
    X_data = normalized_df[feature_cols].values
    y_prices = normalized_df['Close_Original'].values
    dates = normalized_df['Date'].values
    
    X_sequences = []
    y_sequences = []
    sequence_dates = []
    
    # Create sequences
    for i in range(len(X_data) - lookback - max(pred_horizons)):
        # Input: lookback days of features
        X_sequences.append(X_data[i:i+lookback])
        
        # Target: prices at pred_horizons ahead
        y_targets = []
        for horizon in pred_horizons:
            target_idx = i + lookback + horizon - 1
            if target_idx < len(y_prices):
                y_targets.append(y_prices[target_idx])
        
        y_sequences.append(y_targets)
        sequence_dates.append(dates[i + lookback])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Created {len(X_sequences)} sequences")
    print(f"X shape: {X_sequences.shape}")
    print(f"- Sequences: {X_sequences.shape[0]}")
    print(f"- Lookback days: {X_sequences.shape[1]}")
    print(f"- Features: {X_sequences.shape[2]}")
    print(f"y shape: {y_sequences.shape}")
    print(f"- Sequences: {y_sequences.shape[0]}")
    print(f"- Prediction horizons: {y_sequences.shape[1]} (1/7/30 days)")
    
    return X_sequences, y_sequences, sequence_dates

# Create sequences for all three cryptos
print("\n" + "-"*60)
sequence_data['BTC'] = create_sequences(normalized_data['BTC'], "BITCOIN (BTC)")
sequence_data['LTC'] = create_sequences(normalized_data['LTC'], "LITECOIN (LTC)")
sequence_data['ETH'] = create_sequences(normalized_data['ETH'], "ETHEREUM (ETH)")

print("\n" + "="*60)
print("SEQUENCES CREATED")
print("="*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if sequence_data[ticker][0] is not None:
        X, y, dates = sequence_data[ticker]
        print(f"\n{ticker}:")
        print(f"  X_sequences shape: {X.shape}")
        print(f"  y_sequences shape: {y.shape}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")