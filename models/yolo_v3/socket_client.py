import socketio
from base64 import b64decode
from PIL import Image
import io
import numpy as np
import time
import cv2
from detect import YOLO

_model_name = 'yolo_v3'
_last_activity_time = time.time()

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')
    sio.emit('register', _model_name)
    sio.emit('frame_request')

@sio.on("process_frame")
def process_frame(data):
    global _last_activity_time
    start_time = time.time()
    _last_activity_time = start_time
    img = webp_to_img(data)
    # process image
    boxes = yolo.get_prediction(img)
    # build payload
    end_time = time.time()
    _last_activity_time = end_time
    payload = {'name':_model_name,
               'annotations': boxes,
               'clock_time': end_time - start_time}
    sio.emit('frame_complete', payload)
    # request a new frame
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

def poll_timer():
    global _last_activity_time
    while True:
        time.sleep(1)
        if time.time() - _last_activity_time > 5:
            print('No activity. Querying controller again')
            sio.emit('frame_request')

if __name__ == '__main__':
    yolo = YOLO()
    sio.connect('http://localhost:8080')
    poll_timer()
