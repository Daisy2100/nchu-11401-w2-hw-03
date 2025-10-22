"""
Streamlit Web Application for Spam Classification

Interactive web interface for testing spam detection models.
"""

import streamlit as st
import sys
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Run setup on first launch
if not os.path.exists('models/preprocessor.pkl'):
    st.info("🚀 First-time setup: Training models... This may take a few minutes.")
    import setup
    setup.setup()
    st.success("✅ Setup complete! Reloading app...")
    st.rerun()

# Add src to path
sys.path.insert(0, 'src')

from preprocessing import TextPreprocessor
from models import SpamClassifier
from evaluation import plot_confusion_matrix, plot_roc_curve


# Page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models_and_preprocessor():
    """Load all trained models and preprocessor."""
    models = {}
    models_dir = 'models'
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'SVM': 'svm.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
    else:
        preprocessor = None
    
    return models, preprocessor


def main():
    """Main application."""
    
    # Title
    st.title("📧 Email/SMS Spam Classifier")
    st.markdown("**Detect spam messages using Machine Learning**")
    st.markdown("---")
    
    # Load models
    models, preprocessor = load_models_and_preprocessor()
    
    if not models or preprocessor is None:
        st.error("⚠️ Models not found! Please train the models first.")
        st.info("Run: `python src/cli.py train --data data/sms_spam_no_header.csv`")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=0
    )
    
    selected_model = models[model_name]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info(f"**Selected:** {model_name}")
    st.sidebar.markdown(f"**Type:** {selected_model.model_type.replace('_', ' ').title()}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📈 Performance", "ℹ️ About", "📝 Examples"])
    
    # Tab 1: Prediction
    with tab1:
        st.header("Test Message Classification")
        
        # Text input
        message = st.text_area(
            "Enter your message:",
            height=150,
            placeholder="Type or paste a message here to check if it's spam..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            predict_button = st.button("🔍 Classify", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if predict_button and message.strip():
            # Preprocess and predict
            X = preprocessor.transform([message])
            prediction = selected_model.predict(X)[0]
            proba = selected_model.predict_proba(X)[0]
            
            st.markdown("---")
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("### 🚨 SPAM")
                    st.markdown(f"**Confidence:** {proba[1]:.1%}")
                else:
                    st.success("### ✅ HAM (Legitimate)")
                    st.markdown(f"**Confidence:** {proba[0]:.1%}")
            
            with col2:
                # Probability chart
                fig, ax = plt.subplots(figsize=(6, 4))
                categories = ['Ham', 'Spam']
                colors = ['green', 'red']
                ax.bar(categories, proba, color=colors, alpha=0.7)
                ax.set_ylabel('Probability')
                ax.set_ylim([0, 1])
                ax.set_title('Classification Probabilities')
                for i, v in enumerate(proba):
                    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
                st.pyplot(fig)
                plt.close()
        
        elif predict_button:
            st.warning("⚠️ Please enter a message to classify.")
    
    # Tab 2: Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        # Check if outputs exist
        outputs_dir = 'outputs'
        
        if os.path.exists(outputs_dir):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                safe_name = model_name.lower().replace(' ', '_')
                cm_path = os.path.join(outputs_dir, f'confusion_matrix_{safe_name}.png')
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)
                else:
                    st.info("Confusion matrix not available.")
            
            with col2:
                st.subheader("ROC Curve")
                roc_path = os.path.join(outputs_dir, f'roc_curve_{safe_name}.png')
                if os.path.exists(roc_path):
                    st.image(roc_path, use_container_width=True)
                else:
                    st.info("ROC curve not available.")
            
            st.markdown("---")
            
            st.subheader("Feature Importance")
            fi_path = os.path.join(outputs_dir, f'feature_importance_{safe_name}.png')
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
            else:
                st.info("Feature importance visualization not available.")
            
            st.markdown("---")
            
            st.subheader("Model Comparison")
            comp_path = os.path.join(outputs_dir, 'model_comparison.png')
            if os.path.exists(comp_path):
                st.image(comp_path, use_container_width=True)
            else:
                st.info("Model comparison not available.")
        else:
            st.warning("⚠️ Performance visualizations not found. Please train and evaluate models first.")
    
    # Tab 3: About
    with tab3:
        st.header("About This Project")
        
        st.markdown("""
        ### 📚 Overview
        This application demonstrates spam email/SMS classification using multiple machine learning algorithms.
        
        ### 🎯 Purpose
        - **Course:** NCHU IoT Applications and Data Analysis - HW3
        - **Objective:** Build a complete ML pipeline with OpenSpec workflow
        - **Dataset:** SMS Spam Collection (5,500+ messages)
        
        ### 🤖 Models Used
        
        1. **Logistic Regression**
           - Linear model for binary classification
           - Fast training and prediction
           - Good interpretability
        
        2. **Naive Bayes (MultinomialNB)**
           - Probabilistic classifier based on Bayes' theorem
           - Works well with text data
           - Assumes feature independence
        
        3. **Support Vector Machine (SVM)**
           - Finds optimal decision boundary
           - Effective in high-dimensional spaces
           - Robust to overfitting
        
        ### 🔧 Features
        - Text preprocessing (cleaning, tokenization)
        - TF-IDF vectorization
        - Multiple model comparison
        - Real-time spam detection
        - Performance visualizations
        
        ### 📊 Tech Stack
        - **Backend:** Python, scikit-learn, NLTK
        - **Frontend:** Streamlit
        - **Development:** OpenSpec workflow
        
        ### 👨‍💻 GitHub Repository
        [github.com/huanchen1107/2025ML-spamEmail](https://github.com/huanchen1107/2025ML-spamEmail)
        """)
    
    # Tab 4: Examples
    with tab4:
        st.header("Example Messages")
        
        st.markdown("### 📝 Try these examples:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Spam Examples")
            spam_examples = [
                "URGENT! You have won a $1000 Walmart gift card. Go to http://bit.ly/123abc to claim now!",
                "Congratulations! You've been selected for a free iPhone. Call 555-0123 now!",
                "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
                "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
            ]
            
            for i, example in enumerate(spam_examples, 1):
                if st.button(f"Example {i}", key=f"spam_{i}"):
                    st.text_area("Selected message:", example, height=100, key=f"spam_display_{i}")
        
        with col2:
            st.subheader("✅ Ham (Legitimate) Examples")
            ham_examples = [
                "Hi, how are you doing today? Want to grab coffee later?",
                "Meeting rescheduled to 3pm tomorrow in conference room B.",
                "Thanks for your email. I'll get back to you by end of day.",
                "Don't forget to pick up milk on your way home!",
            ]
            
            for i, example in enumerate(ham_examples, 1):
                if st.button(f"Example {i}", key=f"ham_{i}"):
                    st.text_area("Selected message:", example, height=100, key=f"ham_display_{i}")
        
        st.markdown("---")
        st.info("💡 **Tip:** Copy any example above and paste it in the 'Predict' tab to test the classifier!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit | NCHU IoT & Data Analysis 2025"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
