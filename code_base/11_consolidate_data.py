print("="*60)
print("CONSOLIDATING DATA FROM 3 CRYPTOS")
print("="*60)

# Extract data from train_val_test_data dictionary
# Since all 3 cryptos have same split structure, we combine them

# Get BTC data (representative structure)
btc_data = train_val_test_data['BTC']

# Concatenate all 3 cryptos for training
X_train_btc = train_val_test_data['BTC']['X_train']
X_train_ltc = train_val_test_data['LTC']['X_train']
X_train_eth = train_val_test_data['ETH']['X_train']

X_val_btc = train_val_test_data['BTC']['X_val']
X_val_ltc = train_val_test_data['LTC']['X_val']
X_val_eth = train_val_test_data['ETH']['X_val']

X_test_btc = train_val_test_data['BTC']['X_test']
X_test_ltc = train_val_test_data['LTC']['X_test']
X_test_eth = train_val_test_data['ETH']['X_test']

# Concatenate X data
X_train = np.vstack([X_train_btc, X_train_ltc, X_train_eth])
X_val = np.vstack([X_val_btc, X_val_ltc, X_val_eth])
X_test = np.vstack([X_test_btc, X_test_ltc, X_test_eth])

# Concatenate y data
y_train_btc = train_val_test_data['BTC']['y_train']
y_train_ltc = train_val_test_data['LTC']['y_train']
y_train_eth = train_val_test_data['ETH']['y_train']

y_val_btc = train_val_test_data['BTC']['y_val']
y_val_ltc = train_val_test_data['LTC']['y_val']
y_val_eth = train_val_test_data['ETH']['y_val']

y_test_btc = train_val_test_data['BTC']['y_test']
y_test_ltc = train_val_test_data['LTC']['y_test']
y_test_eth = train_val_test_data['ETH']['y_test']

y_train = np.vstack([y_train_btc, y_train_ltc, y_train_eth])
y_val = np.vstack([y_val_btc, y_val_ltc, y_val_eth])
y_test = np.vstack([y_test_btc, y_test_ltc, y_test_eth])

print("\n" + "="*60)
print("DATA CONSOLIDATED FROM 3 CRYPTOS")
print("="*60)

print(f"\nTraining Data:")
print(f"  X_train shape: {X_train.shape}")
print(f"    - Sequences: {X_train.shape[0]} (291×3 from BTC+LTC+ETH)")
print(f"    - Lookback days: {X_train.shape[1]}")
print(f"    - Features: {X_train.shape[2]}")
print(f"  y_train shape: {y_train.shape}")
print(f"    - Sequences: {y_train.shape[0]}")
print(f"    - Horizons: {y_train.shape[1]} (1/7/30-day)")

print(f"\nValidation Data:")
print(f"  X_val shape: {X_val.shape}")
print(f"    - Sequences: {X_val.shape[0]} (62×3 from BTC+LTC+ETH)")
print(f"  y_val shape: {y_val.shape}")

print(f"\nTest Data:")
print(f"  X_test shape: {X_test.shape}")
print(f"    - Sequences: {X_test.shape[0]} (63×3 from BTC+LTC+ETH)")
print(f"  y_test shape: {y_test.shape}")

print("\n✓ ALL DATA READY FOR MODEL TRAINING")
print(f"✓ X_train defined: {X_train.shape}")
print(f"✓ y_train defined: {y_train.shape}")
print(f"✓ X_val defined: {X_val.shape}")
print(f"✓ y_val defined: {y_val.shape}")
print(f"✓ X_test defined: {X_test.shape}")
print(f"✓ y_test defined: {y_test.shape}")
