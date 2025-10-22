# Change Proposal: Add Spam Classification Pipeline

## Why
Build a complete machine learning pipeline for email spam classification to fulfill NCHU ML homework requirements. This extends the Chapter 3 reference implementation with enhanced preprocessing, visualization, and interactive interfaces.

## What Changes
- Add data preprocessing module with text cleaning, tokenization, and TF-IDF vectorization
- Implement multiple ML classifiers (Logistic Regression, Naïve Bayes, SVM)
- Create comprehensive evaluation system with metrics and visualizations
- Build CLI interface for model training and prediction
- Develop Streamlit web interface for interactive demo
- Add configuration management and logging
- Create complete documentation and setup instructions

## Impact
- Affected specs: `spam-classifier` (new capability)
- Affected code: 
  - New: `src/preprocessing.py`, `src/models.py`, `src/evaluation.py`
  - New: `src/cli.py`, `app.py` (Streamlit)
  - New: `config/config.yaml`, `requirements.txt`, `README.md`
- Deployment: Streamlit Cloud at https://2025spamemail.streamlit.app/

## Success Criteria
- [ ] Model achieves >95% accuracy on test set
- [ ] Streamlit demo is deployed and accessible
- [ ] All visualizations render correctly (confusion matrix, ROC curve, feature importance)
- [ ] README provides clear setup and usage instructions
- [ ] Code follows PEP 8 standards
