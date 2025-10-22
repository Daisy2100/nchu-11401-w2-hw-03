# Project Context

## Purpose
Email/SMS Spam Classification project for NCHU "IoT Applications and Data Analysis" course (HW3). This project extends Chapter 3 from "Hands-On Artificial Intelligence for Cybersecurity" with enhanced preprocessing, visualization, and interactive interfaces (CLI/Streamlit) following OpenSpec workflow.

**Core Tasks:**
- Data preprocessing: cleaning, tokenization, TF-IDF vectorization
- Model training: Logistic Regression, Naïve Bayes, SVM
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Interface development: CLI and Streamlit frontend

**Dataset:** sms_spam_no_header.csv from Packt repository

## Tech Stack

**Core:**
- Python 3.10+
- Pandas, NumPy (data processing)
- Scikit-learn (LogisticRegression, MultinomialNB, SVC, TfidfVectorizer)
- NLTK (stopwords, word_tokenize)

**Interfaces:**
- Streamlit (web demo)
- Argparse or Click (CLI)

**Development:**
- Jupyter Notebooks (exploration)
- VS Code
- OpenSpec (specification management)

## Project Conventions

### Code Style
- Follow PEP 8 standards
- Naming: `snake_case` for variables/functions, `PascalCase` for classes
- Use descriptive variable names (e.g., `tfidf_vectorizer`, `spam_classifier`)
- Add docstrings for functions and classes
- Maximum line length: 100 characters

### Architecture Patterns
- **Modular Design**: Core logic in `src/` folder, `app.py` as Streamlit entry point
- **Pipeline Pattern**: Use scikit-learn Pipeline for reproducibility
- **Configuration**: Store hyperparameters in config files
- **Environment**: Use `requirements.txt` for dependency management

### Testing Strategy
- Unit tests for data preprocessing functions
- Integration tests for ML pipeline
- Validation tests for model metrics (accuracy, precision, recall, F1)
- Manual testing through Streamlit UI

### Git Workflow
- Main branch: `main`
- Feature branches: `feature/[feature-name]`
- All changes require OpenSpec proposal first
- Commit messages: Follow conventional commits (e.g., `feat:`, `fix:`, `docs:`)
- GitHub repository: <https://github.com/huanchen1107/2025ML-spamEmail>

## Domain Context

**Problem Type:** Binary classification (ham vs. spam)

**Dataset Structure:**
- File: `sms_spam_no_header.csv`
- Columns: `v1` (label: ham/spam), `v2` (message content)

**Key Challenges:**
- **Imbalanced Data**: Ham messages far outnumber spam
  - Solution: Use F1-score and ROC-AUC instead of accuracy alone
  - Consider stratified sampling
- **Feature Engineering**: Convert raw text to TF-IDF numerical features
- **Text Preprocessing**: Tokenization, stopword removal, normalization

## Important Constraints
- **Academic Project**: NCHU IoT & Data Analysis course HW3
- **Deadline**: 2025-11-05 23:59
- **Required Tools**: Must use OpenSpec and AI CLI for development
- **GitHub Repository**: <https://github.com/huanchen1107/2025ML-spamEmail>
- **Demo Site**: <https://nchu-11401-w2-hw-03-6t6om5uwkkdepcvtibzomu.streamlit.app/>
- **Deployment**: Must have working Streamlit demo accessible online

## External Dependencies
- **Dataset Source**: 
  - <https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv>
- **Reference Repository**: 
  - <https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity>
- **Deployment Platform**: Streamlit Cloud
- **Python Packages**: See `requirements.txt`
