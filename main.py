import json
import torch
import pydub
import numpy as np
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
from time import perf_counter

with open('.json', 'r') as f:
    config = json.load(f)

asr_pipe = config['asr_pipe']

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline("automatic-speech-recognition",
                    model=asr_pipe,
                    max_new_tokens=50,
                    device=device,
                    )

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    t0 = perf_counter()
    file.file.seek = lambda *args: None
    audio = pydub.AudioSegment.from_file(file.file)
    t1 = perf_counter()
    audio, sampling_rate = pydub_to_np(audio)
    audio = audio.mean(1)
    t2 = perf_counter()
    text = pipe({'raw': audio, 'sampling_rate': sampling_rate})['text']
    t3 = perf_counter()
    res = {
        'text': text,
        'processing_times': {
                            'read': t1-t0,
                            'convert_to_numpy': t2-t1,
                            'transcribe': t3-t2,
                            },
        'device': device
    }
    return res


def pydub_to_np(audio):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate
