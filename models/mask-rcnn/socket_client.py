import socketio
import time
from model import OnnxModel
from image_functions import data_url_to_pil

_inactivity_threshold = 5
_last_activity_time = time.time()
_autorun = True

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')
    sio.emit('register', model.name)
    sio.emit('frame_request')


@sio.on("process_frame")
def process_frame(data_url):
    global _last_activity_time
    start_time = time.time()
    _last_activity_time = start_time
    # process image
    img = data_url_to_pil(data_url)
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    payload = model.postprocess(img, output_dict)
    # build payload
    end_time = time.time()
    _last_activity_time = end_time
    print("[INFO] {} processing time: {:.6f} seconds"
          .format(model.name, end_time - start_time))
    sio.emit('frame_complete', payload)
    # request a new frame
    if _autorun:
        sio.emit('frame_request')


@sio.event
def disconnect():
    print('disconnected from server')


def poll_timer():
    global _last_activity_time
    while True:
        time.sleep(1)
        if _autorun and \
                time.time() - _last_activity_time > _inactivity_threshold:
            print('No activity. Querying controller again')
            sio.emit('frame_request')


if __name__ == '__main__':
    model = OnnxModel()
    sio.connect('http://localhost:8080')
    poll_timer()
