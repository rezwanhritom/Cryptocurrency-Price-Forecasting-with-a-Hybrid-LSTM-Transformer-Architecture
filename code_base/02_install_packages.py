# Install all required packages
print("Installing required packages...\n")

!pip install -q ta-lib pandas numpy scikit-learn tensorflow keras-tuner optuna

print("\nAll packages installed successfully!")

print("\nVerifying imports...")

import sklearn
import talib
print(f"scikit-learn: {sklearn.__version__}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"tensorflow: {tf.__version__}")

print("\nAll imports successful!")