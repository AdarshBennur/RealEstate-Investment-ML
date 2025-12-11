"""
Download pre-trained models from cloud storage
This script is called on Streamlit Cloud startup to fetch models
"""

import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination"""
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Downloaded {destination}")

def download_models():
    """
    Download pre-trained models from cloud storage
    
    NOTE: You need to upload your trained models to a cloud storage service
    (Google Drive, Dropbox, GitHub Releases, AWS S3, etc.) and replace the URLs below
    """
    
    models_dir = Path('models')
    
    # Check if models already exist
    if (models_dir / 'classification_model.pkl').exists() and \
       (models_dir / 'regression_model.pkl').exists():
        print("✅ Models already exist")
        return
    
    print("=" * 60)
    print("DOWNLOADING PRE-TRAINED MODELS...")
    print("=" * 60)
    
    # TODO: Replace these URLs with your actual model URLs from cloud storage
    # Example: Google Drive, AWS S3, GitHub Releases, etc.
    
    # For now, show instructions
    print("""
    ⚠️ Models need to be uploaded to cloud storage first!
    
    Steps:
    1. Train models locally
    2. Upload to cloud storage (Google Drive, S3, etc.)
    3. Get direct download URLs
    4. Update this script with URLs
    5. Redeploy to Streamlit Cloud
    """)
    
    # Example (uncomment and replace URLs when ready):
    # download_file(
    #     'https://your-storage.com/classification_model.pkl',
    #     'models/classification_model.pkl'
    # )
    # download_file(
    #     'https://your-storage.com/regression_model.pkl',
    #     'models/regression_model.pkl'
    # )

if __name__ == '__main__':
    download_models()
