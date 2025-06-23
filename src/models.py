
import traceback
from pyabsa import download_all_available_datasets, available_checkpoints
from pyabsa import AspectTermExtraction as ATEPC

# Model Initialization

# 1. Download all required datasets for PyABSA
# This operation might take some time the first time it's run as it downloads
# various datasets used by PyABSA's models.
print("PyABSA: Starting dataset download (if not already present)...")
try:
    download_all_available_datasets()
    print("PyABSA: Datasets download complete.")
except Exception as e:
    print(f"PyABSA: Error during dataset download: {e}")
    print(traceback.format_exc())
    # Continue execution even if download fails, models might still work if data is cached.

# 2. Initialize the Aspect Term Extraction and Polarity Classification (ATEPC) model
# Using 'multilingual' checkpoint for broad language support.
print("\nPyABSA: Initializing Aspect Term Extractor (ATEPC) with 'multilingual' checkpoint...")
try:
    aspect_extractor = ATEPC.AspectExtractor(checkpoint="multilingual")
    print("PyABSA: Aspect Term Extractor initialized successfully.")
except Exception as e:
    print(f"PyABSA: FAILED to initialize Aspect Term Extractor: {e}")
    print(traceback.format_exc())
    aspect_extractor = None

# 3. No ASTE model initialization as requested.

# 4. Print available checkpoints for ATEPC for verification purposes
print("\nPyABSA: Available ATEPC checkpoints:")
try:
    available_checkpoints("ATEPC", True) # Changed to ATEPC specific checkpoints
except Exception as e:
    print(f"PyABSA: Could not retrieve ATEPC checkpoints: {e}")
    print(traceback.format_exc())

