"""
Script to download the SMS Spam dataset.

Downloads the dataset from the Packt GitHub repository.
"""

import urllib.request
import os

DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"
DATASET_PATH = "data/sms_spam_no_header.csv"


def download_dataset():
    """Download the spam dataset if it doesn't exist."""
    
    if os.path.exists(DATASET_PATH):
        print(f"✅ Dataset already exists at: {DATASET_PATH}")
        return
    
    print(f"📥 Downloading dataset from: {DATASET_URL}")
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        
        # Download file
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        
        print(f"✅ Dataset downloaded successfully to: {DATASET_PATH}")
        
        # Check file size
        file_size = os.path.getsize(DATASET_PATH)
        print(f"📊 File size: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"\n💡 You can manually download from:")
        print(f"   {DATASET_URL}")
        print(f"   and save it to: {DATASET_PATH}")


if __name__ == "__main__":
    download_dataset()
