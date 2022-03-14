from __future__ import annotations
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dialect_detection.engine import Model


app = FastAPI()
model=Model()


class TestingData(BaseModel):
    texts: List[str]

class PredictionObject(BaseModel):
    text: str
    prediction: str

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]

@app.post("/predict-ml", summary="predict a batch of sentences with machine learning model")
def predict_ml(testing_data:TestingData):

    try:
        predictions = model.predict_ml(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict-dl", summary="predict a batch of sentences with Deep learning model")
def predict_dl(testing_data:TestingData):
    try:
        predictions = model.predict_dl(testing_data.texts)
        return PredictionsObject(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))



@app.get("/")
def home():
    return({"message": "System is Ready!!!"})


