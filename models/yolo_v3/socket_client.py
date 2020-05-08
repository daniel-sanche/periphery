import socketio
from base64 import b64decode
from PIL import Image
import io
import numpy as np
import time
from model import YOLO

_model_name = 'yolo_v3'
_inactivity_threshold = 5
_last_activity_time = time.time()
_autorun = True

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')
    sio.emit('register', _model_name)
    sio.emit('frame_request')


@sio.on("process_frame")
def process_frame(data_url):
    global _last_activity_time
    start_time = time.time()
    _last_activity_time = start_time
    # process image
    img = _decode_img(data_url)
    boxes = yolo.get_prediction(img)
    # build payload
    end_time = time.time()
    _last_activity_time = end_time
    print("[INFO] {} processing time: {:.6f} seconds"
          .format(_model_name, end_time - start_time))
    payload = {'name': _model_name,
               'annotations': boxes,
               'clock_time': end_time - start_time}
    sio.emit('frame_complete', payload)
    # request a new frame
    if _autorun:
        sio.emit('frame_request')


@sio.event
def disconnect():
    print('disconnected from server')


def _decode_img(data_url):
    image_data = b64decode(data_url.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    npimg = np.array(img)
    image = npimg.copy()
    return image


def poll_timer():
    global _last_activity_time
    while True:
        time.sleep(1)
        if _autorun and \
                time.time() - _last_activity_time > _inactivity_threshold:
            print('No activity. Querying controller again')
            sio.emit('frame_request')


if __name__ == '__main__':
    yolo = YOLO()
    sio.connect('http://localhost:8080')
    poll_timer()
