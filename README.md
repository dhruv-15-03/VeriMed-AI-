# VeriMed-AI

**VeriMed-AI** is a professional-grade AI-powered toolkit for medical risk prediction, focused on cardiovascular and diabetes risk assessment from patient records and medical reports. It leverages state-of-the-art machine learning models to provide accurate and actionable insights for healthcare professionals, researchers, and developers.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [AI Models](#ai-models)
- [Installation](#installation)
- [Usage](#usage)
  - [Heart Disease Risk](#heart-disease-risk)
  - [Diabetes Risk](#diabetes-risk)
- [Data Preprocessing](#data-preprocessing)
- [Input Formats](#input-formats)
- [Outputs](#outputs)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

VeriMed-AI enables robust medical analytics using AI. It can process patient datasets and medical reports (PDF, DOCX, TXT) to predict the risk of heart attack and diabetes. The project uses ensemble models, advanced text extraction, and preprocessing to handle real-world health data.

---

## Features

- **AI-Powered Risk Prediction:** Predicts heart attack and diabetes risk using machine learning.
- **Automated Text Extraction:** Reads and parses patient reports in PDF, DOCX, and TXT formats.
- **Data Preprocessing:** Cleans and normalizes medical data, encoding categorical and scaling numeric features.
- **Model Interpretability:** Prints prediction probabilities and key metrics for transparency.
- **Extensible Pipeline:** Modular scripts for easy adaptation to new conditions or datasets.

---

## AI Models

- **Heart Disease Risk Model:**  
  - Algorithm: Random Forest Classifier
  - Inputs: Demographics, cholesterol, BP, diabetes, family history, lifestyle factors, etc.
  - Location: `models/heart_disease_rf_model.pkl`
  - Trained with preprocessed CSV datasets, categorical encoding, and standard scaling.

- **Diabetes Risk Model:**
  - Algorithm: Random Forest (with probability calibration using CalibratedClassifierCV)
  - Inputs: Pregnancies, glucose, BP, skin thickness, insulin, BMI, pedigree, age
  - Location: `models/diabetes_model.pkl`
  - Handles imbalanced data with SMOTE-Tomek, scales features, and outputs calibrated probabilities.

- **Text Extraction & NLP:**  
  - Uses Spacy and PyPDF2 for extracting and parsing medical reports.
  - Extracts relevant metrics for prediction.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruv-15-03/VeriMed-AI-.git
   cd VeriMed-AI-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   *(Required libraries: pandas, scikit-learn, imbalanced-learn, joblib, spacy, PyPDF2, etc.)*

3. **Download Spacy models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Usage

### Heart Disease Risk

- **Train model:**
  ```bash
  python script.py <cleaned_csv>
  ```
- **Predict from report:**
  ```bash
  python script.py <report.pdf>
  ```

### Diabetes Risk

- **Train model:**
  ```bash
  python Diabetes_script.py <diabetes_csv>
  ```
- **Predict from report:**
  ```bash
  python Diabetes_script.py <report.pdf>
  ```

---

## Data Preprocessing

- **General CSV:**  
  Use `preprocess.py` to clean and encode raw health datasets.
- **Diabetes CSV:**  
  Use `diabetes_preprocess.py` for diabetes-specific datasets.

Example (general):
```bash
python preprocess.py <raw_data.csv>
```
Example (diabetes):
```bash
python diabetes_preprocess.py <raw_diabetes.csv>
```

---

## Input Formats

- **Patient Datasets:** CSV files with required columns for each condition.
- **Reports:** PDF, DOCX, or TXT containing patient metrics.

---

## Outputs

- **Model files:** Saved in `models/` directory.
- **Predictions:** Console output of risk classification and probability.
- **Cleaned datasets:** Saved as CSV after preprocessing.

---

## File Structure

```
VeriMed-AI-/
│
├── script.py                # Heart disease pipeline
├── Diabetes_script.py       # Diabetes pipeline
├── preprocess.py            # General preprocessing
├── diabetes_preprocess.py   # Diabetes-specific preprocessing
├── Extract_Text.py          # Text extraction from reports
├── models/                  # Saved trained models
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---



## Contact

- **Author:** [dhruv-15-03](https://github.com/dhruv-15-03)
- **Repository:** [VeriMed-AI-](https://github.com/dhruv-15-03/VeriMed-AI-)

---

*For any queries, suggestions, or support, please open an issue in this repository.*
