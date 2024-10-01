from fastapi import FastAPI, UploadFile
from PIL import Image
import numpy as np
from models.attention_unet import attention_unet

app = FastAPI()

model = attention_unet()
model.load_weights('weights/attention_unet_weights.h5')

@app.post("/predict")
async def predict(file: UploadFile):
    """API endpoint to get metastasis segmentation predictions."""
    img = np.array(Image.open(file.file).convert('L')) 
    img = np.expand_dims(img, axis=[0, -1]) 
    pred = model.predict(img)
    return {"segmentation": pred.tolist()}
