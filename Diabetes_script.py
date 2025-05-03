import os, sys, joblib, pandas as pd, re

import PyPDF2
import spacy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
from sklearn.calibration import CalibratedClassifierCV
from Extract_Text import extract_text

nlp = spacy.load("en_core_web_sm")
def predict_from_report1(file_path: str, save_path: str = "output.txt") -> str:
    extracted_text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text += page.extract_text() or ""

    with open(save_path, 'w', encoding='utf-8') as out:
        out.write(extracted_text)

    return extracted_text
def extract_metrics(text: str) -> dict:
    patterns = {
        'Pregnancies': r'(?i)pregnancies\s*[:\-]\s*(\d+)',
        'Glucose':     r'(?i)glucose\s*[:\-]\s*(\d+)',
        'BloodPressure': r'(?i)blood pressure\s*[:\-]\s*(\d+)',
        'SkinThickness': r'(?i)skin thickness\s*[:\-]\s*(\d+)',
        'Insulin':       r'(?i)insulin\s*[:\-]\s*(\d+)',
        'BMI':           r'(?i)bmi\s*[:\-]\s*(\d+\.?\d*)',
        'DiabetesPedigreeFunction': r'(?i)diabetes pedigree function\s*[:\-]\s*(\d+\.?\d*)',
        'Age':           r'(?i)age\s*[:\-]\s*(\d+)'
    }
    data = {}
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            data[k] = float(v) if '.' in v else int(v)
    return data

def train_and_save_diabetes(csv_path, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna()
    features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']
    X, y = df[features], df['Diabetes'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)  # SMOTE-Tomek :contentReference[oaicite:9]{index=9}

    scaler = StandardScaler().fit(X_res)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    X_res_scaled = scaler.transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    base_clf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42)
    clf = CalibratedClassifierCV(base_clf, cv=5)  # probability calibration :contentReference[oaicite:10]{index=10}
    clf.fit(X_res_scaled, y_res)

    acc = clf.score(X_test_scaled, y_test)
    print(f'Test Accuracy: {acc:.3f}')

    joblib.dump(clf, os.path.join(model_dir, 'diabetes_model.pkl'))
    print("Model saved.")

def predict_from_report(path):
    d1=predict_from_report1(path)
    text = extract_text("output.txt")
    metrics = extract_metrics(text)
    scaler = joblib.load('models/scaler.pkl')
    clf    = joblib.load('models/diabetes_model.pkl')

    feat = list(scaler.feature_names_in_)
    row = [metrics.get(f, 0) for f in feat]
    X = scaler.transform([row])
    prob = clf.predict_proba(X)[0,1]
    thresh = 0.3
    pred = int(prob >= thresh)
    print(f'Probability: {prob:.2f}, Prediction: {"Diabetic" if pred else "Non-Diabetic"}')
    return f'Probability: {prob:.2f}, Prediction: {"Diabetic" if pred else "Non-Diabetic"}'

if __name__=='__main__':
    if len(sys.argv)<2:
        print("Usage: python script.py <data.csv> or <report.txt>")
        sys.exit(1)
    arg = sys.argv[1]
    if arg.lower().endswith('.csv'):
        train_and_save_diabetes(arg)
    else:
        predict_from_report(arg)