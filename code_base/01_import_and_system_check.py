import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check GPUs available
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
if len(gpus) > 0:
    print(f"  GPU Details: {gpus[0]}")
    print("GPU IS READY TO USE!")
else:
    print("NO GPU DETECTED - CHECK SETTINGS!")

# Check CPUs
cpus = tf.config.list_physical_devices('CPU')
print(f"CPUs available: {len(cpus)}")

print("\n" + "="*60)
print("SYSTEM CONFIGURATION COMPLETE")
print("="*60)