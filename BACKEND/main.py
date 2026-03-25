from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model import model_list,inference
from typing import Literal
import os


labels = ['CAT', 'DOG']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"],
    allow_credentials = True
)



@app.post("/Comparison")
async def compare(file : UploadFile = File(...),model_name:Literal["VGG","ResNet","EfficientNet"] = Form(...)):
    file_location = f"temp_{file.filename}"

    with open(file_location,"wb") as buffer:
        buffer.write(await file.read())

    if(model_name == "VGG"):
        model = model_list[0]
    elif(model_name == "ResNet"):
        model = model_list[1]
    elif(model_name == "EfficientNet"):
        model = model_list[2]

    pred_class,conf = inference(model,file_location)
    os.remove(file_location)
    return{
        "pred_label":labels[pred_class],
        "conf":conf*100
    }
    

