from fastapi import FastAPI, File,Path,HTTPException,Query, UploadFile
from pydantic import BaseModel,Field,model_validator,computed_field
from typing import Annotated,Optional,Literal
import torch
from model import MNISTModel
from io import BytesIO
import torch.nn as nn
from PIL import Image
import numpy as np
app = FastAPI()

@app.get("/")
def hello():
    return {'message': 'welcome to digit recognition api'}
@app.post("/predict")
async def predict(image: UploadFile = File(..., description="Image file in bytes format")):

    contents = await image.read() 
    model = MNISTModel()
    model.load_state_dict(torch.load('mnist_model1.pth'))
    model.eval()

    img = Image.open(BytesIO(contents)).convert('L') 
    img = img.resize((28, 28))  
    img_array = np.array(img)
    img_array = img_array.astype(np.float32)

    a = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
    a = a.reshape(1, 1, 28, 28)
    logits = model(a)
    ans = torch.argmax(logits, dim=1)
    confidence = logits[0,ans].item()
    
    return {"predicted_digit": ans.item(),"confidence": confidence}
