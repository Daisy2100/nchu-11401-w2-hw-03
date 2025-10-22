# Email Spam Classification with ML

🎓 **NCHU IoT & Data Analysis Course - HW3**

A complete machine learning pipeline for email/SMS spam classification, following OpenSpec specification-driven development workflow.

## 📊 Project Overview

This project implements a binary classification system to detect spam messages using multiple ML algorithms:
- Logistic Regression
- Naïve Bayes (MultinomialNB)
- Support Vector Machine (SVM)

### Features
- ✅ Text preprocessing (cleaning, tokenization, TF-IDF vectorization)
- ✅ Multiple classifier comparison
- ✅ Comprehensive evaluation metrics
- ✅ Interactive Streamlit web interface
- ✅ Command-line interface (CLI)
- ✅ Visualization (confusion matrix, ROC curves, feature importance)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/huanchen1107/2025ML-spamEmail.git
cd 2025ML-spamEmail
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

4. Place the dataset:
Download `sms_spam_no_header.csv` and place it in the `data/` folder.

### Usage

#### CLI Interface
```bash
# Train models
python src/cli.py train --data data/sms_spam_no_header.csv

# Predict single message
python src/cli.py predict --text "Congratulations! You've won a prize!"

# Evaluate models
python src/cli.py evaluate
```

#### Streamlit Web Interface
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
.
├── src/                    # Core source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── models.py           # ML models
│   ├── evaluation.py       # Evaluation & metrics
│   └── cli.py              # CLI interface
├── data/                   # Dataset folder
├── models/                 # Trained models (.pkl)
├── outputs/                # Visualizations & reports
├── notebooks/              # Jupyter notebooks
├── config/                 # Configuration files
├── openspec/               # OpenSpec documentation
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Naïve Bayes | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD |

## 🌐 Demo

Live demo: [https://nchu-11401-w2-hw-03-6t6om5uwkkdepcvtibzomu.streamlit.app/](https://nchu-11401-w2-hw-03-6t6om5uwkkdepcvtibzomu.streamlit.app/)

## 📚 Dataset

- **Source**: [Packt AI for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv)
- **Format**: CSV with columns `v1` (label) and `v2` (message)
- **Size**: ~5,500 messages
- **Classes**: ham (legitimate), spam

## 🛠️ Development

This project follows [OpenSpec](https://openspec.dev/) workflow:
- See `openspec/changes/add-spam-classification-pipeline/` for the full specification
- All changes are tracked through proposals and tasks

## 📝 License

Educational project for NCHU course. 

## 👥 Author

- **Student ID**: [Your ID]
- **Name**: [Your Name]
- **Course**: IoT Applications and Data Analysis
- **Assignment**: HW3
- **Deadline**: 2025-11-05

## 🙏 Acknowledgments

- Based on Chapter 3 of "Hands-On Artificial Intelligence for Cybersecurity" by Packt
- Dataset from SMS Spam Collection
