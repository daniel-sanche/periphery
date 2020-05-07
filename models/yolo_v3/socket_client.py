import socketio
from base64 import b64decode
from PIL import Image
import io
import numpy as np
import time
import cv2
from detect import YOLO

_model_name = 'yolo_v3'

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')
    sio.emit('register', _model_name)
    sio.emit('frame_request')

@sio.on("process_frame")
def my_message(data):
    print('frame received')
    t = time.time()
    img = webp_to_img(data)

    # process image
    boxes = yolo.get_prediction(img)
    elapsed_time = time.time() - t
    payload = {'name':_model_name,
               'annotations': boxes,
               'process_time': elapsed_time}
    print(elapsed_time)
    sio.emit('frame_complete', payload)
    sio.emit('frame_request')

@sio.event
def disconnect():
    print('disconnected from server')

def webp_to_img(blob):
    image_data = b64decode(blob.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

yolo = YOLO()
sio.connect('http://localhost:8080')

sio.wait()
