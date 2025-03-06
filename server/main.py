import io
import uvicorn
from fileinput import filename
from typing import Annotated, Literal
import cv2
from fastapi import FastAPI,File,UploadFile, Form,Cookie, WebSocket, WebSocketDisconnect
from PIL import Image
from fastapi.responses import JSONResponse
import numpy as np
from enum import Enum
import requests
import time
from fastapi.middleware.cors import CORSMiddleware
from semantic_search import semantic_search
from pathlib import Path



app = FastAPI()

websockets = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search")
async def semantic_search_(image: UploadFile,query:Annotated[str, Form()]):
    
    image =  await image.read()
    image = Image.open(io.BytesIO(image))
    sm_se = semantic_search(image)
    res = sm_se.infer(query)
    return res
if __name__=="__main__":
   uvicorn.run("main:app",host='0.0.0.0', port=8081, reload=False, workers=1)
    
   
