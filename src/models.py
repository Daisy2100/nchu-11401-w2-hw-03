"""
Model Training Module for Spam Classification

This module implements multiple classification models for spam detection.
"""

import pickle
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import os


class SpamClassifier:
    """Wrapper class for spam classification models."""
    
    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of model ('logistic', 'naive_bayes', 'svm')
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.is_trained = False
        
    def _create_model(self, **kwargs):
        """Create the specified model."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                C=kwargs.get('C', 1.0),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(
                alpha=kwargs.get('alpha', 1.0)
            )
        elif self.model_type == 'svm':
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'linear'),
                random_state=kwargs.get('random_state', 42),
                probability=True  # Enable probability estimates
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'SpamClassifier':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Self for method chaining
        """
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"{self.model_type} model trained successfully!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.model.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1
            return proba
    
    def save_model(self, file_path: str):
        """
        Save model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {file_path}")
    
    @staticmethod
    def load_model(file_path: str) -> 'SpamClassifier':
        """
        Load model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded SpamClassifier instance
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {file_path}")
        return model


def train_all_models(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    config: Dict[str, Any] = None
) -> Dict[str, SpamClassifier]:
    """
    Train all available models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary for models
        
    Returns:
        Dictionary mapping model names to trained classifiers
    """
    if config is None:
        config = {
            'logistic_regression': {'C': 1.0, 'max_iter': 1000, 'random_state': 42},
            'naive_bayes': {'alpha': 1.0},
            'svm': {'C': 1.0, 'kernel': 'linear', 'random_state': 42}
        }
    
    models = {}
    
    # Train Logistic Regression
    print("\n" + "="*50)
    lr = SpamClassifier('logistic', **config.get('logistic_regression', {}))
    lr.train(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # Train Naive Bayes
    print("\n" + "="*50)
    nb = SpamClassifier('naive_bayes', **config.get('naive_bayes', {}))
    nb.train(X_train, y_train)
    models['Naive Bayes'] = nb
    
    # Train SVM
    print("\n" + "="*50)
    svm = SpamClassifier('svm', **config.get('svm', {}))
    svm.train(X_train, y_train)
    models['SVM'] = svm
    
    print("\n" + "="*50)
    print("All models trained successfully!")
    
    return models


def save_all_models(models: Dict[str, SpamClassifier], output_dir: str = 'models'):
    """
    Save all trained models.
    
    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
    """
    for name, model in models.items():
        # Create safe filename
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(output_dir, filename)
        model.save_model(filepath)


if __name__ == "__main__":
    # Example usage
    print("Testing models module...")
    
    # Create dummy data
    from sklearn.datasets import make_classification
    X_train, y_train = make_classification(
        n_samples=1000, 
        n_features=100, 
        n_classes=2, 
        random_state=42
    )
    
    # Train a single model
    clf = SpamClassifier('logistic')
    clf.train(X_train, y_train)
    
    # Make predictions
    predictions = clf.predict(X_train[:10])
    print(f"\nSample predictions: {predictions}")
