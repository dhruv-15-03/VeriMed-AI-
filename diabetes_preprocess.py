import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_diabetes_csv(input_file: str, output_file: str = "cleaned_diabetes_dataset1.csv") -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df = df.dropna()
    scaler = MinMaxScaler()
    feature_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    df.to_csv(output_file, index=False)
    print(f"Cleaned and normalized diabetes dataset saved as {output_file}")

    return df
