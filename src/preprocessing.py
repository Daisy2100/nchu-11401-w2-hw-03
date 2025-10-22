"""
Data Preprocessing Module for Spam Classification

This module handles loading, cleaning, and transforming text data for spam classification.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Tuple, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline for spam classification."""
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95):
        """
        Initialize the preprocessor.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation, converting to lowercase, etc.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Cleaned text string
            
        Returns:
            Text with stopwords removed
        """
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            return ' '.join(filtered_tokens)
        except Exception as e:
            print(f"Error in stopword removal: {e}")
            return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Full preprocessing pipeline for a single text.
        
        Args:
            text: Raw text string
            
        Returns:
            Fully preprocessed text
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        # Transform texts to TF-IDF features
        features = self.vectorizer.fit_transform(processed_texts)
        
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform texts to TF-IDF features
        features = self.vectorizer.transform(processed_texts)
        
        return features


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the spam dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with columns 'label' and 'message'
    """
    try:
        # Load CSV with custom column names
        df = pd.read_csv(file_path, encoding='latin-1', header=None, names=['label', 'message'])
        
        # Convert labels to binary (0 = ham, 1 = spam)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Remove any NaN values
        df = df.dropna()
        
        print(f"Dataset loaded: {len(df)} messages")
        print(f"Ham: {sum(df['label'] == 0)}, Spam: {sum(df['label'] == 1)}")
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets with stratification.
    
    Args:
        df: DataFrame with 'label' and 'message' columns
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # Maintain class distribution
    )
    
    print(f"Training set: {len(train_df)} messages")
    print(f"Test set: {len(test_df)} messages")
    
    return train_df, test_df


if __name__ == "__main__":
    # Example usage
    print("Testing preprocessing module...")
    
    # Test text preprocessing
    preprocessor = TextPreprocessor()
    sample_text = "FREE! Win a £1000 cash prize! Call now: www.example.com"
    cleaned = preprocessor.preprocess_text(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned: {cleaned}")
