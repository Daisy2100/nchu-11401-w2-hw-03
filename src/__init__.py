"""
Spam Classification Package

Core modules for email/SMS spam detection using machine learning.
"""

__version__ = "1.0.0"
__author__ = "NCHU Student"

from .preprocessing import TextPreprocessor, load_dataset, prepare_train_test_split
from .models import SpamClassifier, train_all_models, save_all_models
from .evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models,
    plot_feature_importance
)

__all__ = [
    'TextPreprocessor',
    'load_dataset',
    'prepare_train_test_split',
    'SpamClassifier',
    'train_all_models',
    'save_all_models',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'compare_models',
    'plot_feature_importance',
]
