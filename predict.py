import torch
from model import ToxinMLPRegressor

model = ToxinMLPRegressor(num_features=448).double()
model.load_state_dict(torch.load('ToxinModel.pth'))
model.eval()

def predict_toxin(data):
    pred = model(data.double()).squeeze().detach().numpy()
    pred = pred * 3.5
    pred = 10 ** pred
    return pred