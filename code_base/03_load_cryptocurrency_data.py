print("="*60)
print("LOADING ALL THREE CRYPTOCURRENCIES")
print("="*60)

import os
import glob

# Cryptocurrency Timeseries 2020 dataset path
crypto_path = '/kaggle/input/cryptocurrency-timeseries-2020'

print(f"\nDataset path: {crypto_path}")

if os.path.exists(crypto_path):
    # List all files
    all_files = os.listdir(crypto_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for i, f in enumerate(csv_files, 1):
        file_size = os.path.getsize(os.path.join(crypto_path, f)) / 1024 / 1024
        print(f"  {i}. {f} ({file_size:.1f} MB)")
else:
    print(f"Dataset not found at: {crypto_path}")
    print("Make sure to add 'cryptocurrency-timeseries-2020' as input in your Kaggle notebook")

print("\n" + "="*60)
print("LOADING THREE CRYPTOCURRENCIES FROM GEMINI EXCHANGE")
print("="*60)

# Dictionary to store dataframes
crypto_data = {}

# Load Bitcoin
print("\nLoading BITCOIN (BTC) from Gemini...")
try:
    btc_file = os.path.join(crypto_path, 'gemini_BTCUSD_2020_1min.csv')
    df_btc = pd.read_csv(btc_file)
    crypto_data['BTC'] = df_btc
    print(f"Loaded: gemini_BTCUSD_2020_1min.csv")
    print(f"Shape: {df_btc.shape}")
    print(f"Columns: {df_btc.columns.tolist()}")
except FileNotFoundError:
    print(f"File not found: gemini_BTCUSD_2020_1min.csv")
    crypto_data['BTC'] = None
except Exception as e:
    print(f"Error loading BTC: {e}")
    crypto_data['BTC'] = None

# Load Litecoin
print("\nLoading LITECOIN (LTC) from Gemini...")
try:
    ltc_file = os.path.join(crypto_path, 'gemini_LTCUSD_2020_1min.csv')
    df_ltc = pd.read_csv(ltc_file)
    crypto_data['LTC'] = df_ltc
    print(f"Loaded: gemini_LTCUSD_2020_1min.csv")
    print(f"Shape: {df_ltc.shape}")
    print(f"Columns: {df_ltc.columns.tolist()}")
except FileNotFoundError:
    print(f"File not found: gemini_LTCUSD_2020_1min.csv")
    crypto_data['LTC'] = None
except Exception as e:
    print(f"Error loading LTC: {e}")
    crypto_data['LTC'] = None

# Load Ethereum
print("\nLoading ETHEREUM (ETH) from Gemini...")
try:
    eth_file = os.path.join(crypto_path, 'gemini_ETHUSD_2020_1min.csv')
    df_eth = pd.read_csv(eth_file)
    crypto_data['ETH'] = df_eth
    print(f"Loaded: gemini_ETHUSD_2020_1min.csv")
    print(f"Shape: {df_eth.shape}")
    print(f"Columns: {df_eth.columns.tolist()}")
except FileNotFoundError:
    print(f"File not found: gemini_ETHUSD_2020_1min.csv")
    crypto_data['ETH'] = None
except Exception as e:
    print(f"Error loading ETH: {e}")
    crypto_data['ETH'] = None

print("\n" + "="*60)
print("MULTI-CRYPTO DATASET SUMMARY")
print("="*60)

datasets_info = []

for ticker in ['BTC', 'LTC', 'ETH']:
    if ticker in crypto_data and crypto_data[ticker] is not None:
        df = crypto_data[ticker]
        datasets_info.append((ticker, df.shape))
        print(f"\n{ticker}:")
        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]}")
        print(f"Exchange: Gemini")
        print(f"Period: 2020")
        print(f"Interval: 1-minute")
    else:
        print(f"\n{ticker}: NOT LOADED")

print(f"\nTOTAL DATASETS LOADED: {len(datasets_info)}/3")
print(f"STRATEGY: Multi-Crypto Analysis (Gemini Exchange, 2020)")
print(f"DATA SOURCE: cryptocurrency-timeseries-2020")