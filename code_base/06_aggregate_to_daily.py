print("\n" + "="*60)
print("AGGREGATING 1-MINUTE → DAILY DATA")
print("="*60)

# Dictionary to store daily aggregated data
daily_data = {}

def aggregate_to_daily(df, ticker_name):
    print(f"\nAggregating {ticker_name} to daily...")
    
    if df is None:
        print(f"{ticker_name} data is None, skipping")
        return None
    
    df = df.copy()
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        print(f"No Date column found, skipping {ticker_name}")
        return None
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date only (remove time)
    df['DateOnly'] = df['Date'].dt.date
    
    # Aggregate to daily OHLCV
    daily = df.groupby('DateOnly').agg({
        'Open': 'first',      # First open of the day
        'High': 'max',        # Highest price of the day
        'Low': 'min',         # Lowest price of the day
        'Close': 'last',      # Last close of the day
        'Volume': 'sum'       # Total volume of the day
    }).reset_index()
    
    # Convert DateOnly back to datetime
    daily['Date'] = pd.to_datetime(daily['DateOnly'])
    daily = daily.drop('DateOnly', axis=1)
    
    print(f"Aggregated {len(df):,} rows → {len(daily)} daily candles")
    print(f"Date range: {daily['Date'].min().date()} to {daily['Date'].max().date()}")
    
    return daily

# Aggregate all three cryptos
print("\n" + "-"*60)
daily_data['BTC'] = aggregate_to_daily(crypto_data['BTC'], "BITCOIN (BTC)")
daily_data['LTC'] = aggregate_to_daily(crypto_data['LTC'], "LITECOIN (LTC)")
daily_data['ETH'] = aggregate_to_daily(crypto_data['ETH'], "ETHEREUM (ETH)")

print("\n" + "="*60)
print("DAILY DATA AGGREGATED")
print("="*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if daily_data[ticker] is not None:
        df = daily_data[ticker]
        print(f"\n{ticker}:")
        print(f"  Days: {len(df)}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"  First day: Open=${df['Open'].iloc[0]:.2f}, Close=${df['Close'].iloc[0]:.2f}")
        print(f"  Last day: Open=${df['Open'].iloc[-1]:.2f}, Close=${df['Close'].iloc[-1]:.2f}")
