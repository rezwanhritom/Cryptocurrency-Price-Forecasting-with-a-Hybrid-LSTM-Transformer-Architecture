print("="*60)
print("DATA CLEANING & PREPARATION")
print("="*60)

# Function to clean cryptocurrency data
def clean_crypto_data(df, name):
    print(f"\nCleaning {name}...")
    
    if df is None:
        print(f"{name} data is None, skipping")
        return None
    
    df = df.copy()
    
    # Handle different date column names
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp']
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['Date'] = df['date']
    else:
        # Create date from index if available
        try:
            df['Date'] = pd.to_datetime(df.index)
        except:
            print(f"Could not convert date - no date column found")
            return None
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(f"No missing values")
    else:
        print(f"Missing values found:")
        for col in df.columns:
            if missing[col] > 0:
                print(f"      {col}: {missing[col]}")
    
    # Remove duplicates
    initial = len(df)
    df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
    removed = initial - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    
    print(f"Final shape: {df.shape}")
    
    return df

# Clean all three cryptos
print("\n" + "-"*60)
crypto_data['BTC'] = clean_crypto_data(crypto_data['BTC'], "BITCOIN (BTC)")
crypto_data['LTC'] = clean_crypto_data(crypto_data['LTC'], "LITECOIN (LTC)")
crypto_data['ETH'] = clean_crypto_data(crypto_data['ETH'], "ETHEREUM (ETH)")

print("\n" + "="*60)
print("SUMMARY: ALL THREE CRYPTOS CLEANED")
print("="*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if ticker in crypto_data and crypto_data[ticker] is not None:
        df = crypto_data[ticker]
        print(f"\n{ticker}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        if 'Close' in df.columns:
            print(f"  Close price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    else:
        print(f"\n{ticker}:NOT LOADED")

print("\nDATA CLEANING COMPLETE")