import socketio
from base64 import b64decode
from PIL import Image
import io
import numpy as np

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.on("stream")
def my_message(data):
    print('message received')
    img = webp_to_img(data)
    print(img.shape)

@sio.event
def disconnect():
    print('disconnected from server')

def webp_to_img(blob):
    image_data = b64decode(blob.split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    return np.asarray(image)

sio.connect('http://localhost:8080')
sio.wait()
