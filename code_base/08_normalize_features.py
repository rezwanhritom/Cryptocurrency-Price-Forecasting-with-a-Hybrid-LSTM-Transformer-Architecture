print("="*60)
print("NORMALIZING FEATURES TO [0, 1]")
print("="*60)

from sklearn.preprocessing import MinMaxScaler

# Dictionary to store normalized data and scalers
normalized_data = {}
scalers = {}

def normalize_features(features_df, ticker_name):
    print(f"\nNormalizing features for {ticker_name}...")
    
    if features_df is None:
        print(f"{ticker_name} data is None, skipping")
        return None, None
    
    df = features_df.copy()
    
    # Keep Date and Close separate
    dates = df['Date'].values
    close_prices = df['Close'].values
    
    # Select only feature columns (exclude Date and Close)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
    feature_values = df[feature_cols].values
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform
    normalized_features = scaler.fit_transform(feature_values)
    
    # Create normalized dataframe
    normalized_df = pd.DataFrame(normalized_features, columns=feature_cols)
    normalized_df['Date'] = dates
    normalized_df['Close_Original'] = close_prices
    
    print(f"   ✓ Normalized {len(feature_cols)} features")
    print(f"   ✓ Shape: {normalized_features.shape}")
    print(f"   ✓ All features now in range [0, 1]")
    
    return normalized_df, scaler

# Normalize all three cryptos
print("\n" + "-"*60)
normalized_data['BTC'], scalers['BTC'] = normalize_features(features_data['BTC'], "BITCOIN (BTC)")
normalized_data['LTC'], scalers['LTC'] = normalize_features(features_data['LTC'], "LITECOIN (LTC)")
normalized_data['ETH'], scalers['ETH'] = normalize_features(features_data['ETH'], "ETHEREUM (ETH)")

print("\n" + "="*60)
print("FEATURES NORMALIZED")
print("="*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if normalized_data[ticker] is not None:
        df = normalized_data[ticker]
        feature_cols = [col for col in df.columns if col not in ['Date', 'Close_Original']]
        print(f"\n{ticker}:")
        print(f"  Shape: {df.shape}")
        print(f"  Features: {len(feature_cols)} normalized indicators")
        print(f"  Range: [0, 1]")
        print(f"  Sample values (last day):")
        print(f"    SMA_20: {df['SMA_20'].iloc[-1]:.4f}")
        print(f"    RSI_14: {df['RSI_14'].iloc[-1]:.4f}")
        print(f"    MACD: {df['MACD'].iloc[-1]:.4f}")