print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Create statistics for all three cryptos
crypto_stats = {}

for ticker in ['BTC', 'LTC', 'ETH']:
    if ticker not in crypto_data or crypto_data[ticker] is None:
        continue
    
    df = crypto_data[ticker]
    
    print(f"\n{'='*60}")
    print(f"{ticker} PRICE STATISTICS (2020)")
    print(f"{'='*60}")
    
    close_prices = df['Close'].dropna()
    daily_returns = df['Close'].pct_change() * 100
    daily_returns = daily_returns.dropna()
    
    stats = {
        'min': close_prices.min(),
        'max': close_prices.max(),
        'mean': close_prices.mean(),
        'median': close_prices.median(),
        'std': close_prices.std(),
        'return_mean': daily_returns.mean(),
        'return_std': daily_returns.std(),
        'return_max': daily_returns.max(),
        'return_min': daily_returns.min(),
    }
    
    crypto_stats[ticker] = {
        'df': df,
        'close_prices': close_prices,
        'daily_returns': daily_returns,
        'stats': stats
    }
    
    print(f"  Min price: ${stats['min']:.2f}")
    print(f"  Max price: ${stats['max']:.2f}")
    print(f"  Mean price: ${stats['mean']:.2f}")
    print(f"  Median price: ${stats['median']:.2f}")
    print(f"  Std dev: ${stats['std']:.2f}")
    
    print(f"\n{ticker} RETURNS STATISTICS")
    print(f"  Mean return: {stats['return_mean']:.6f}% (per minute)")
    print(f"  Std dev: {stats['return_std']:.4f}%")
    print(f"  Max gain: {stats['return_max']:.4f}%")
    print(f"  Max loss: {stats['return_min']:.4f}%")
    
    if 'Volume' in df.columns:
        print(f"\n{ticker} VOLUME STATISTICS")
        print(f"  Mean volume: {df['Volume'].mean():.2f}")
        print(f"  Total volume: {df['Volume'].sum():.2e}")

print("\n" + "="*60)
print("GENERATING MULTI-CRYPTO VISUALIZATIONS")
print("="*60)

# Create 3x3 grid: one row per crypto, 3 columns (price, returns, histogram)
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Multi-Crypto EDA (2020): Bitcoin, Litecoin, Ethereum - Gemini Exchange', 
             fontsize=16, fontweight='bold', y=0.995)

tickers = ['BTC', 'LTC', 'ETH']

for row_idx, ticker in enumerate(tickers):
    if ticker not in crypto_stats or crypto_stats[ticker] is None:
        # Skip if data not available
        for col_idx in range(3):
            axes[row_idx, col_idx].text(0.5, 0.5, f'{ticker}: Data not available',
                                        ha='center', va='center',
                                        transform=axes[row_idx, col_idx].transAxes)
            axes[row_idx, col_idx].set_title(f'{ticker}', fontweight='bold')
        continue
    
    df = crypto_stats[ticker]['df']
    close_prices = crypto_stats[ticker]['close_prices']
    daily_returns = crypto_stats[ticker]['daily_returns']
    
    # Subsample for faster plotting (keep every Nth point)
    sample_rate = max(1, len(df) // 500)
    df_plot = df.iloc[::sample_rate].copy()
    
    # Column 0: Price over time
    axes[row_idx, 0].plot(df_plot['Date'], df_plot['Close'], 
                          linewidth=1.5, color='#2E86AB')
    axes[row_idx, 0].fill_between(df_plot['Date'], df_plot['Close'], 
                                   alpha=0.3, color='#2E86AB')
    axes[row_idx, 0].set_title(f'{ticker} - Price Over Time', fontweight='bold')
    axes[row_idx, 0].set_ylabel('Price (USD)')
    axes[row_idx, 0].grid(alpha=0.3)
    axes[row_idx, 0].tick_params(axis='x', rotation=45)
    
    # Column 1: Daily returns
    returns_plot = daily_returns.iloc[::sample_rate].reset_index(drop=True)
    dates_for_returns = df_plot['Date'][1:].reset_index(drop=True)
    min_len = min(len(returns_plot), len(dates_for_returns))
    
    axes[row_idx, 1].plot(dates_for_returns[:min_len], returns_plot[:min_len],
                          linewidth=0.5, color='#A23B72')
    axes[row_idx, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[row_idx, 1].set_title(f'{ticker} - Returns', fontweight='bold')
    axes[row_idx, 1].set_ylabel('Return (%)')
    axes[row_idx, 1].grid(alpha=0.3)
    axes[row_idx, 1].tick_params(axis='x', rotation=45)
    
    # Column 2: Return distribution
    axes[row_idx, 2].hist(daily_returns, bins=50, edgecolor='black', 
                          alpha=0.7, color='#F18F01')
    axes[row_idx, 2].set_title(f'{ticker} - Return Distribution', fontweight='bold')
    axes[row_idx, 2].set_xlabel('Return (%)')
    axes[row_idx, 2].set_ylabel('Frequency')
    axes[row_idx, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('01_multi_crypto_eda_2020.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_multi_crypto_eda_2020.png")
plt.show()

for ticker in tickers:
    if ticker in crypto_stats and crypto_stats[ticker] is not None:
        df = crypto_stats[ticker]['df']
        print(f"\n{ticker}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

print(f"All three cryptos: BTC, LTC, ETH prepared and analyzed")
print(f"Data source: cryptocurrency-timeseries-2020 (Gemini Exchange, 2020)\n")