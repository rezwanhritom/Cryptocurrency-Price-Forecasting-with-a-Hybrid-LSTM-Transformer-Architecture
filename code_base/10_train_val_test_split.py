print("="*60)
print("TEMPORAL TRAIN/VALIDATION/TEST SPLIT")
print("="*60)

# Dictionary to store split data
train_val_test_data = {}

def temporal_train_val_test_split(X_seq, y_seq, dates, ticker_name, 
                                  train_ratio=0.70, val_ratio=0.15):
    """
    Split data temporally (respecting time order, no data leakage)
    
    train_ratio: 70% for training
    val_ratio: 15% for validation
    test_ratio: 15% for testing (implicit)
    """
    print(f"\nSplitting {ticker_name} data...")
    
    total_samples = len(X_seq)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # Temporal split (maintain chronological order)
    X_train = X_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    
    y_train = y_seq[:train_size]
    y_val = y_seq[train_size:train_size+val_size]
    y_test = y_seq[train_size+val_size:]
    
    dates_train = dates[:train_size]
    dates_val = dates[train_size:train_size+val_size]
    dates_test = dates[train_size+val_size:]
    
    print(f"   ✓ Total sequences: {total_samples}")
    print(f"   ✓ Training: {train_size} sequences ({train_ratio*100:.0f}%)")
    print(f"     - Date range: {dates_train[0]} to {dates_train[-1]}")
    print(f"   ✓ Validation: {val_size} sequences ({val_ratio*100:.0f}%)")
    print(f"     - Date range: {dates_val[0]} to {dates_val[-1]}")
    print(f"   ✓ Test: {test_size} sequences ({(1-train_ratio-val_ratio)*100:.0f}%)")
    print(f"     - Date range: {dates_test[0]} to {dates_test[-1]}")
    
    print(f"\nIMPORTANT: Temporal split maintains chronological order")
    print(f"     - No data leakage")
    print(f"     - Realistic evaluation")
    print(f"     - Test data is future unseen by model")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'dates_train': dates_train, 'dates_val': dates_val, 'dates_test': dates_test
    }

# Split all three cryptos
print("\n" + "-"*60)
for ticker in ['BTC', 'LTC', 'ETH']:
    if sequence_data[ticker][0] is not None:
        X, y, dates = sequence_data[ticker]
        train_val_test_data[ticker] = temporal_train_val_test_split(X, y, dates, ticker)

print("\n" + "="*60)
print("TRAIN/VAL/TEST SPLIT DONE")
print("="*60)

print("\nFINAL DATA SUMMARY:")
print("-"*60)

for ticker in ['BTC', 'LTC', 'ETH']:
    if ticker in train_val_test_data:
        data = train_val_test_data[ticker]
        print(f"\n{ticker}:")
        print(f"  X_train shape: {data['X_train'].shape}")
        print(f"  X_val shape: {data['X_val'].shape}")
        print(f"  X_test shape: {data['X_test'].shape}")
        print(f"  y_train shape: {data['y_train'].shape}")
        print(f"  y_val shape: {data['y_val'].shape}")
        print(f"  y_test shape: {data['y_test'].shape}")