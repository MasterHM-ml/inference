import cv2
import yaml
import base64
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, Form

from model_api import output_image, parse_label

with open("config.yaml", "r") as ff:
    config = yaml.load(ff, yaml.SafeLoader)

app = FastAPI()

@app.post(path=f"/{config['app']['endpoint']}")
async def inference(image: UploadFile, confidence: float = Form(...)):
    img_read = await image.read()
    img = cv2.imdecode(np.frombuffer(base64.b64decode(img_read), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = output_image(img, confidence) # get inference from model
    if list(results[0].boxes):
        return parse_label(results[0])
    else:
        return {"class_name": False}

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["app"]["port"])
