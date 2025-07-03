import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error

def load_data():
    df=pd.read_csv('../input/breast-cancer-metabric/Breast Cancer METABRIC.csv') #reading the dataset
    df=df.dropna(subset=['Overall Survival (Months)', 'Overall Survival Status', 'Tumor Stage'])
    df['Overall Survival Status']=df['Overall Survival Status'].map({'Living':1,'Deceased':0})
    df['ER Status']=df['ER Status'].map({'Yes':1,'No':0})
    return df
df=load_data()
def train_models(df):
    X = df[['Age at Diagnosis','Tumor Size','Neoplasm Histologic Grade','ER Status','Nottingham prognostic index','Lymph nodes examined positive']]
    y_stage = df['Tumor Stage']
    y_survival = np.log1p(df['Overall Survival (Months)'])
    y_event = df['Overall Survival Status']
    X_s1, X_s2, y_s1, y_s2 = train_test_split(X, y_stage, test_size=0.2, random_state=42)
    X_t1, X_t2, y_t1, y_t2 = train_test_split(X, y_survival, test_size=0.2, random_state=42)
    X_e1, X_e2, y_e1, y_e2 = train_test_split(X, y_event, test_size=0.2, random_state=42)
    pipe_stage = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', DecisionTreeClassifier(random_state=42))
    ]) #so this imputes the data wherever it is missing, 
    pipe_stage.fit(X_s1, y_s1)
    pipe_survival = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),  # XGBoost doesn't need this, but OK to keep for consistency
    ('model', XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    pipe_survival.fit(X_t1,y_t1)
    pipe_event = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_event.fit(X_e1, y_e1)
    return pipe_stage,pipe_survival,pipe_event
pipe_stage,pipe_survival_pipe_event=train_models(df)

st.title("BREAST CANCER PREDICTIONS BASED ON ML")
st.sidebar.header("Enter Patient Data")

age = st.sidebar.number_input("Age at Diagnosis", 20, 100, 50)
tumor_size = st.sidebar.number_input("Tumor Size (mm)", 0, 200, 30)
grade = st.sidebar.selectbox("Histologic Grade", [1, 2, 3])
er_status = st.sidebar.selectbox("ER Status", [0, 1], format_func=lambda x: "Yes" if x else "No")
npi = st.sidebar.number_input("Nottingham Prognostic Index", 0.0, 10.0, 3.0)
lymph = st.sidebar.number_input("Lymph Nodes Examined Positive", 0, 50, 1)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([{
        'Age at Diagnosis': age,
        'Tumor Size': tumor_size,
        'Neoplasm Histologic Grade': grade,
        'ER Status': er_status,
        'Nottingham prognostic index': npi,
        'Lymph nodes examined positive': lymph
    }])

    pred_stage = pipe_stage.predict(input_df)[0]
    pred_survival_log = pipe_survival.predict(input_df)[0]
    pred_survival = np.expm1(pred_survival_log)  # antilog
    pred_event = pipe_event.predict(input_df)[0]

    st.write("## Predictions")
    st.write(f"**Tumor Stage:** Stage {int(pred_stage)}")
    st.write(f"**Estimated Survival Time:** {pred_survival:.1f} months")
    st.write(f"**Recurrence/Death Prediction:** {'Deceased' if pred_event == 0 else 'Living'}")

    
