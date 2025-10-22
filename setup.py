"""
Setup script for first-time Streamlit deployment.

Downloads dataset and trains models if they don't exist.
"""

import os
import sys

def setup():
    """Run first-time setup."""
    
    # Check if models exist
    models_dir = 'models'
    model_files = ['logistic_regression.pkl', 'naive_bayes.pkl', 'svm.pkl', 'preprocessor.pkl']
    
    models_exist = all(
        os.path.exists(os.path.join(models_dir, f)) for f in model_files
    )
    
    if models_exist:
        print("✅ Models already exist, skipping training...")
        return
    
    print("🚀 First-time setup: Training models...")
    
    # Download dataset if needed
    dataset_path = 'data/sms_spam_no_header.csv'
    if not os.path.exists(dataset_path):
        print("📥 Downloading dataset...")
        import download_dataset
        download_dataset.download_dataset()
    
    # Train models
    print("🤖 Training models (this may take a few minutes)...")
    sys.path.insert(0, 'src')
    from cli import train_command
    
    class Args:
        data = dataset_path
        config = 'config/config.yaml'
    
    train_command(Args())
    print("✅ Setup complete!")


if __name__ == "__main__":
    setup()
