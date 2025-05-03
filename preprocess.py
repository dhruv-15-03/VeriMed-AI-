import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_csv(input_file: str, output_file: str = "cleaned_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(input_file)
    df = df.dropna()

    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic BP'] = bp_split[0].astype(float)
    df['Diastolic BP'] = bp_split[1].astype(float)
    df = df.drop(columns=['Blood Pressure'])

    label_encoders = {}
    categorical_columns = [
        'Sex', 'Family History', 'Smoking', 'Obesity',
        'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
        'Previous Heart Problems', 'Medication Use', 'Stress Level',
        'Sedentary Hours Per Day', 'Income', 'Country', 'Continent', 'Hemisphere'
    ]
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = MinMaxScaler()
    numeric_columns = [
        'Age', 'Cholesterol', 'Heart Rate', 'Diabetes', 'BMI',
        'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
        'Systolic BP', 'Diastolic BP', 'Heart Attack Risk'
    ]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df.to_csv(output_file, index=False)
    print(f"Cleaned and normalized dataset saved as {output_file}")

    return df
