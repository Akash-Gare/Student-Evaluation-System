from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI(title="Student Performance AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # sab frontend allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../model/model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "../model/preprocessor.pkl"))

# Pydantic schema
class StudentInput(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API"}

@app.post("/predict")
def predict(data: StudentInput):
    df = pd.DataFrame([data.dict()])
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": float(max(proba))
    }
