import uvicorn
import logging

import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

from preprocess import preprocess_pipeline
from model import ToxinMLPRegressor  
from predict import predict_toxin

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="api.log",
    filemode="a",
)
log = logging.getLogger(__name__)

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame(data["samples"])
    features = df.drop("hsi_id", axis=1)

    try:
        X = preprocess_pipeline(features)
        log.info("Sample preprocessed.")
    except Exception as e:
        print("Error during preprocessing:", e)
        log.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        pred = predict_toxin(X)
        log.info(f"Prediction made: {pred}")
        return {"hsiId": df["hsi_id"],
                "prediction": pred}
    except Exception as e:
        print("Error during prediction:", e)
        log.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))            
    
if __name__ == "__main__":
    uvicorn.run(app, port=5000)