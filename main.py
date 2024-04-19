import tensorflow as tf
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import json
import librosa
from urllib.request import urlopen
import pydub
import requests

MODEL = tf.keras.models.load_model('cnn_csv_99.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]

@app.get('/')
async def index(): 
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(filepath: str = Body(...)):
    try:
        # Predict Emotion
        data = json.loads(filepath)

        wav = io.BytesIO()
        with urlopen(data["URL"]) as response:
            response.seek = lambda *args: None
            pydub.AudioSegment.from_file(response).export(wav, "mp3")
        wav.seek(0)
        raw_audio, _ = librosa.load(wav, sr=44100, duration=3)

        
        mfccs = librosa.feature.mfcc(y=raw_audio, sr=44100, n_mfcc=13)
        mfccs = np.mean(mfccs, axis=1)
        # if len(mfccs) < 130:
        #     mfccs = np.pad(mfccs, (0, 130 - len(mfccs)), mode='constant')
        # elif len(mfccs) > 130:
        #     mfccs = mfccs[:130]

        reshaped_mfccs = mfccs.reshape(1, 13)

        preds = np.argmax(MODEL.predict([reshaped_mfccs]))

        # Generate suggestions
        API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        output = query({
            "inputs": f"My baby is feeling {classes[preds]}. What can I do as a parent/caregiver?\n",
        })
        print(output[0]['generated_text'])

        return {
            "response": classes[preds],
            "output": output[0]['generated_text']
        }
    except Exception as e:
        print(e)
        return {
            "response": "Problem finding the emotion."
        }