# Design Document: Spam Classification Pipeline

## Context
This project implements an email/SMS spam classification system for educational purposes, extending the Packt "Hands-On AI for Cybersecurity" reference with enhanced features. The system must handle text preprocessing, model training, evaluation, and provide both CLI and web interfaces.

## Goals / Non-Goals

### Goals
- Build reproducible ML pipeline using scikit-learn
- Support multiple classification algorithms for comparison
- Provide rich visualizations for model interpretability
- Create user-friendly interfaces (CLI + Streamlit)
- Ensure code quality and documentation standards
- Deploy working demo to Streamlit Cloud

### Non-Goals
- Real-time email integration
- Production-grade scalability
- Deep learning models (keep it simple with traditional ML)
- Multi-language support (English only)
- User authentication/multi-user support

## Decisions

### Decision 1: Modular Architecture
**What**: Separate concerns into distinct modules (preprocessing, models, evaluation, visualization)

**Why**: 
- Maintainability: Easy to modify individual components
- Testability: Can unit test each module independently
- Reusability: Components can be used in different contexts

**Alternatives considered**:
- Monolithic script: Rejected due to poor maintainability
- Notebook-only: Rejected as harder to deploy

### Decision 2: TF-IDF for Feature Extraction
**What**: Use TfidfVectorizer for converting text to numerical features

**Why**:
- Industry standard for text classification
- Good balance of simplicity and effectiveness
- Works well with traditional ML algorithms

**Alternatives considered**:
- Count Vectorizer: Less effective than TF-IDF
- Word embeddings: Overkill for this dataset size

### Decision 3: Three Classifier Comparison
**What**: Implement Logistic Regression, Naïve Bayes, and SVM

**Why**:
- Demonstrates different ML approaches
- Educational value in comparing performance
- All proven effective for text classification

**Alternatives considered**:
- Single model: Less educational value
- Deep learning: Too complex for this dataset

### Decision 4: Streamlit for Web UI
**What**: Use Streamlit for interactive demo

**Why**:
- Rapid development (minimal code)
- Free hosting on Streamlit Cloud
- Python-native (no JavaScript needed)
- Great for ML demos

**Alternatives considered**:
- Flask: More setup required
- React: Too much complexity

## Risks / Trade-offs

### Risk 1: Dataset Imbalance
- **Risk**: SMS spam dataset may be imbalanced (more ham than spam)
- **Mitigation**: Calculate class distribution, potentially use stratified splitting

### Risk 2: Overfitting on Small Dataset
- **Risk**: Model may overfit if dataset is small
- **Mitigation**: Use cross-validation, monitor train vs. test metrics

### Risk 3: Streamlit Deployment Limits
- **Risk**: Free Streamlit Cloud has resource limits
- **Mitigation**: Optimize model size, use caching

## Migration Plan

### Phase 1: Core Pipeline (Week 1)
1. Set up project structure
2. Implement data preprocessing
3. Train and evaluate models
4. Verify metrics meet success criteria (>95% accuracy)

### Phase 2: Interfaces (Week 2)
5. Build CLI interface
6. Develop Streamlit app
7. Add visualizations

### Phase 3: Deployment & Documentation (Week 3)
8. Deploy to Streamlit Cloud
9. Complete documentation
10. Final testing and refinement

### Rollback Strategy
- Keep dataset in version control for reproducibility
- Document model hyperparameters for retraining
- Streamlit Cloud allows easy rollback to previous commits

## Open Questions
- Should we implement cross-validation or just train/test split?
  - **Decision**: Start with simple train/test split (80/20), add CV if time permits
- Do we need a config file or hardcode parameters?
  - **Decision**: Use config.yaml for flexibility
- Should we persist trained models to disk?
  - **Decision**: Yes, save models as .pkl files for reuse
