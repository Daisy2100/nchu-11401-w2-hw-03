"""
Command-Line Interface for Spam Classification

Provides commands for training, prediction, and evaluation.
"""

import argparse
import sys
import os
import yaml
import pickle
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor, load_dataset, prepare_train_test_split
from models import train_all_models, save_all_models, SpamClassifier
from evaluation import (
    evaluate_model, print_classification_report, 
    plot_confusion_matrix, plot_roc_curve, compare_models, 
    plot_feature_importance
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}


def train_command(args):
    """Train spam classification models."""
    print("="*60)
    print("SPAM CLASSIFICATION - TRAINING MODE")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths
    data_path = args.data or config.get('data', {}).get('dataset_path', 'data/sms_spam_no_header.csv')
    output_dir = config.get('output', {}).get('models_dir', 'models')
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    df = load_dataset(data_path)
    
    # Split data
    test_size = config.get('data', {}).get('test_size', 0.2)
    random_state = config.get('data', {}).get('random_state', 42)
    train_df, test_df = prepare_train_test_split(df, test_size, random_state)
    
    # Preprocess text
    print("\nPreprocessing text...")
    max_features = config.get('preprocessing', {}).get('max_features', 5000)
    min_df = config.get('preprocessing', {}).get('min_df', 2)
    max_df = config.get('preprocessing', {}).get('max_df', 0.95)
    
    preprocessor = TextPreprocessor(max_features=max_features, min_df=min_df, max_df=max_df)
    
    X_train = preprocessor.fit_transform(train_df['message'].tolist())
    X_test = preprocessor.transform(test_df['message'].tolist())
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    os.makedirs(output_dir, exist_ok=True)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Train models
    models_config = config.get('models', {})
    models = train_all_models(X_train, y_train, models_config)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, name)
        results[name] = metrics
        print_classification_report(y_test, y_pred, name)
    
    # Save models
    save_all_models(models, output_dir)
    
    # Create visualizations
    outputs_dir = config.get('output', {}).get('outputs_dir', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test)
        safe_name = name.lower().replace(' ', '_')
        plot_confusion_matrix(
            y_test, y_pred, name, 
            save_path=os.path.join(outputs_dir, f'confusion_matrix_{safe_name}.png')
        )
    
    # ROC curves
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        safe_name = name.lower().replace(' ', '_')
        plot_roc_curve(
            y_test, y_proba, name,
            save_path=os.path.join(outputs_dir, f'roc_curve_{safe_name}.png')
        )
    
    # Model comparison
    compare_models(results, save_path=os.path.join(outputs_dir, 'model_comparison.png'))
    
    # Feature importance
    for name, model in models.items():
        safe_name = name.lower().replace(' ', '_')
        plot_feature_importance(
            preprocessor.vectorizer, model, top_n=20,
            save_path=os.path.join(outputs_dir, f'feature_importance_{safe_name}.png')
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved in: {output_dir}/")
    print(f"Visualizations saved in: {outputs_dir}/")


def predict_command(args):
    """Predict whether a message is spam or ham."""
    print("="*60)
    print("SPAM CLASSIFICATION - PREDICTION MODE")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    models_dir = config.get('output', {}).get('models_dir', 'models')
    
    # Determine model file
    if args.model:
        model_path = os.path.join(models_dir, f'{args.model}.pkl')
    else:
        model_path = os.path.join(models_dir, 'logistic_regression.pkl')
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        model = SpamClassifier.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        print("Please train models first using: python src/cli.py train")
        return
    
    # Load preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Get message to predict
    if args.text:
        message = args.text
    else:
        print("\nEnter message to classify (or 'quit' to exit):")
        message = input("> ")
    
    if message.lower() == 'quit':
        return
    
    # Preprocess and predict
    X = preprocessor.transform([message])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Message: {message}")
    print(f"\nPrediction: {'SPAM' if prediction == 1 else 'HAM'}")
    print(f"Confidence: {max(proba):.2%}")
    print(f"  - Ham probability: {proba[0]:.2%}")
    print(f"  - Spam probability: {proba[1]:.2%}")


def evaluate_command(args):
    """Evaluate trained models on test set."""
    print("="*60)
    print("SPAM CLASSIFICATION - EVALUATION MODE")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load test data
    data_path = args.data or config.get('data', {}).get('dataset_path', 'data/sms_spam_no_header.csv')
    print(f"\nLoading dataset from: {data_path}")
    df = load_dataset(data_path)
    
    # Split to get test set (using same parameters as training)
    test_size = config.get('data', {}).get('test_size', 0.2)
    random_state = config.get('data', {}).get('random_state', 42)
    _, test_df = prepare_train_test_split(df, test_size, random_state)
    
    # Load preprocessor
    models_dir = config.get('output', {}).get('models_dir', 'models')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Preprocess test data
    X_test = preprocessor.transform(test_df['message'].tolist())
    y_test = test_df['label'].values
    
    # Load and evaluate all models
    model_files = ['logistic_regression.pkl', 'naive_bayes.pkl', 'svm.pkl']
    model_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
    
    results = {}
    for model_file, model_name in zip(model_files, model_names):
        model_path = os.path.join(models_dir, model_file)
        
        try:
            model = SpamClassifier.load_model(model_path)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred, model_name)
            print_classification_report(y_test, y_pred, model_name)
            results[model_name] = metrics
        except FileNotFoundError:
            print(f"Warning: Model not found at {model_path}")
    
    # Summary comparison
    if results:
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        import pandas as pd
        df_results = pd.DataFrame(results).T
        print(df_results.to_string())


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spam Email Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train spam classification models')
    train_parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict if a message is spam')
    predict_parser.add_argument('--text', type=str, help='Message text to classify')
    predict_parser.add_argument('--model', type=str, 
                               choices=['logistic_regression', 'naive_bayes', 'svm'],
                               help='Model to use for prediction')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
