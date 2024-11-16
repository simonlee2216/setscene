# config.py

import warnings
import os

# Suppress FutureWarnings (e.g., from the transformers library)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress TensorFlow-related warnings (only show errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def apply_configurations():
    print("Environment configurations applied. Warnings suppressed.")
