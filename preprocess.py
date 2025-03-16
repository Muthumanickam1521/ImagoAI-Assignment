import joblib
import torch

global scaler
scaler = joblib.load('ToxinScaler.pkl')

def preprocess_pipeline(data):
    assert data.shape[1] == 448

    if data.isna().sum().sum()>0:
        return None
    
    X = data.values
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float64)
    return X