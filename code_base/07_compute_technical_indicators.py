print("="*60)
print("COMPUTING 15 TECHNICAL INDICATORS")
print("="*60)

import numpy as np
from scipy.stats import linregress

# Dictionary to store feature data
features_data = {}

def compute_technical_indicators(daily_df, ticker_name):
    print(f"\nComputing indicators for {ticker_name}...")
    
    if daily_df is None:
        print(f"{ticker_name} data is None, skipping")
        return None
    
    df = daily_df.copy().reset_index(drop=True)
    
    # Initialize features dataframe
    features = pd.DataFrame()
    features['Date'] = df['Date']
    features['Close'] = df['Close']
    
    # ===== TREND INDICATORS =====
    # 1. Simple Moving Average (20-day)
    features['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # 2. Simple Moving Average (50-day)
    features['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    # 3. Exponential Moving Average (12-day)
    features['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    
    # 4. Exponential Moving Average (26-day)
    features['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # ===== MOMENTUM INDICATORS =====
    # 5. Relative Strength Index (14-day)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    features['RSI_14'] = calculate_rsi(df['Close'].values, period=14)
    
    # 6. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = exp1 - exp2
    
    # 7. MACD Signal Line
    features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
    
    # ===== VOLATILITY INDICATORS =====
    # 8. Average True Range (14-day)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    # 9. Bollinger Bands Upper
    sma20 = df['Close'].rolling(window=20, min_periods=1).mean()
    std20 = df['Close'].rolling(window=20, min_periods=1).std()
    features['Bollinger_Upper'] = sma20 + (std20 * 2)
    
    # 10. Bollinger Bands Lower
    features['Bollinger_Lower'] = sma20 - (std20 * 2)
    
    # ===== VOLUME INDICATORS =====
    # 11. On-Balance Volume (OBV)
    obv = np.zeros(len(df))
    obv[0] = 0
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    features['OBV'] = obv
    
    # ===== CUSTOM FEATURES =====
    # 12. Daily Return (%)
    features['Daily_Return'] = ((df['Close'] - df['Open']) / df['Open'] * 100)
    
    # 13. Volatility (20-day standard deviation)
    features['Volatility_20'] = df['Close'].rolling(window=20, min_periods=1).std()
    
    # 14. Price Momentum (Close / SMA_20)
    features['Price_Momentum'] = df['Close'] / features['SMA_20']
    
    # 15. Volume SMA (5-day moving average of volume)
    features['Volume_SMA_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    
    # Handle NaN values created by indicators (fill with forward fill)
    features = features.fillna(method='bfill').fillna(method='ffill')
    
    print(f"Computed 15 indicators")
    print(f"Features shape: {features.shape}")
    print(f"Indicators: SMA_20, SMA_50, EMA_12, EMA_26, RSI_14, MACD,")
    print(f"MACD_Signal, ATR_14, Bollinger_Upper/Lower, OBV,")
    print(f"Daily_Return, Volatility_20, Price_Momentum, Volume_SMA_5")
    
    return features

# Compute indicators for all three cryptos
print("\n" + "-"*60)
features_data['BTC'] = compute_technical_indicators(daily_data['BTC'], "BITCOIN (BTC)")
features_data['LTC'] = compute_technical_indicators(daily_data['LTC'], "LITECOIN (LTC)")
features_data['ETH'] = compute_technical_indicators(daily_data['ETH'], "ETHEREUM (ETH)")

print("\n" + "="*60)
print("TECHNICAL INDICATORS COMPUTED")
print("="*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if features_data[ticker] is not None:
        df = features_data[ticker]
        print(f"\n{ticker}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  First indicator values:")
        print(f"    SMA_20: ${df['SMA_20'].iloc[-1]:.2f}")
        print(f"    RSI_14: {df['RSI_14'].iloc[-1]:.2f}")
        print(f"    MACD: {df['MACD'].iloc[-1]:.6f}")