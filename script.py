import os
import sys
import re
import joblib
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from Extract_Text import extract_text
import PyPDF2


def predict_from_report1(file_path: str, save_path: str = "output.txt") -> str:
    extracted_text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text += page.extract_text() or ""

    with open(save_path, 'w', encoding='utf-8') as out:
        out.write(extracted_text)

    return extracted_text



def load_nlp_model():
    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        return spacy.load("en_core_web_sm")

nlp = load_nlp_model()


def extract_metrics(text: str) -> dict:
    patterns = {
        'Age': r'(?:age)[:\s]*(\d{1,3})',
        'Sex': r'(?:sex|gender)[:\s]*(male|female)',
        'Cholesterol': r'(?:cholesterol)[:\s]*(\d{2,4})',
        'Blood Pressure': r'(?:blood pressure|bp)[:\s]*(\d{2,3}/\d{2,3})',
        'Heart Rate': r'(?:heart rate|hr)[:\s]*(\d{2,3})',
        'Diabetes': r'(?:diabetes)[:\s]*(yes|no|1|0)',
        'Family History': r'(?:family history)[:\s]*(yes|no|1|0)',
        'Smoking': r'(?:smoking)[:\s]*(yes|no|1|0)',
        'Obesity': r'(?:obesity)[:\s]*(yes|no|1|0)',
        'Alcohol Consumption': r'(?:alcohol consumption)[:\s]*(\d+\.?\d*)',
        'Exercise Hours Per Week': r'(?:exercise hours per week)[:\s]*(\d+\.?\d*)',
        'Stress Level': r'(?:stress level)[:\s]*(\d+\.?\d*)',
        'Sedentary Hours Per Day': r'(?:sedentary hours per day)[:\s]*(\d+\.?\d*)',
        'Income': r'(?:income)[:\s]*(\d+)',
        'BMI': r'(?:bmi)[:\s]*(\d+\.?\d*)',
        'Triglycerides': r'(?:triglycerides)[:\s]*(\d+)',
        'Physical Activity Days Per Week': r'(?:physical activity days per week)[:\s]*(\d+)',
        'Sleep Hours Per Day': r'(?:sleep hours per day)[:\s]*(\d+\.?\d*)',
        'Diet': r'(?:diet)[:\s]*(poor|average|good)',
        'Previous Heart Problems': r'(?:previous heart problems)[:\s]*(yes|no|1|0)',
        'Medication Use': r'(?:medication use)[:\s]*(yes|no|1|0)'
    }
    metrics = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1).lower() if key in ['Sex','Diabetes','Family History','Smoking','Obesity','Diet','Previous Heart Problems','Medication Use'] else m.group(1)
            metrics[key] = val
    bp = metrics.pop('Blood Pressure', None)
    if bp:
        try:
            s, d = bp.split('/')
            metrics['Systolic BP'] = float(s)
            metrics['Diastolic BP'] = float(d)
        except:
            pass
    return metrics


def train_and_save(csv_path: str, model_dir: str = 'models'):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna()
    if 'Blood Pressure' in df.columns:
        bp = df['Blood Pressure'].str.split('/', expand=True)
        df['Systolic BP'] = bp[0].astype(float)
        df['Diastolic BP'] = bp[1].astype(float)
        df.drop('Blood Pressure', axis=1, inplace=True)
    df.drop(['Patient ID','Country','Continent','Hemisphere'], axis=1, inplace=True, errors='ignore')
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk'].astype(int)
    cat_cols = ['Sex','Diabetes','Family History','Smoking','Obesity','Diet','Previous Heart Problems','Medication Use']
    enc_dir = os.path.join(model_dir,'encoders')
    os.makedirs(enc_dir, exist_ok=True)
    for col in cat_cols:
        X[col] = X[col].astype(str).str.lower()
        le = LabelEncoder().fit(X[col])
        X[col] = le.transform(X[col])
        joblib.dump(le, os.path.join(enc_dir, f"{col}.pkl"))
    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, os.path.join(model_dir,'scaler.pkl'))
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    print(f'Test Accuracy: {acc:.3f}')
    joblib.dump(clf, os.path.join(model_dir,'heart_disease_rf_model.pkl'))
    print(f'Model saved to {model_dir}/heart_disease_rf_model.pkl')

SCALER_PATH = 'models/scaler.pkl'
MODEL_PATH  = 'models/heart_disease_rf_model.pkl'

def predict_from_report(file_path: str):
    d1=predict_from_report1(file_path)
    text = extract_text("output.txt")
    metrics = extract_metrics(text)
    defaults = {'Age':50,'Sex':'male','Cholesterol':200,'Systolic BP':120,'Diastolic BP':80,'Heart Rate':70,
                'Diabetes':'no','Family History':'no','Smoking':'no','Obesity':'no','Alcohol Consumption':'0',
                'Exercise Hours Per Week':'2','Stress Level':'5','Sedentary Hours Per Day':'6','Income':'50000',
                'BMI':'25','Triglycerides':'150','Physical Activity Days Per Week':'3','Sleep Hours Per Day':'7',
                'Diet':'average','Previous Heart Problems':'no','Medication Use':'no'}
    scaler = joblib.load(SCALER_PATH)
    feature_cols = list(scaler.feature_names_in_)
    row = {col: metrics.get(col, defaults.get(col, 0)) for col in feature_cols}
    df_in = pd.DataFrame([row], columns=feature_cols)
    cat_cols = ['Sex','Diabetes','Family History','Smoking','Obesity','Diet','Previous Heart Problems','Medication Use']
    for col in cat_cols:
        le = joblib.load(os.path.join('models','encoders',f"{col}.pkl"))
        df_in[col] = df_in[col].astype(str).str.lower()
        df_in[col] = df_in[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df_in[col] = le.transform(df_in[col])
    X_scaled = scaler.transform(df_in)
    clf = joblib.load(MODEL_PATH)
    pred = clf.predict(X_scaled)[0]
    print(f'Prediction: {pred}')
    return str(pred)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <cleaned_csv> or <report>")
        sys.exit(1)
    arg = sys.argv[1]
    if arg.lower().endswith('.csv'):
        train_and_save(arg)
    else:
        predict_from_report(arg)
