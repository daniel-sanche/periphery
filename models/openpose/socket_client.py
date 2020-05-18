import socketio
import time
from model import Model
from image_functions import data_url_to_pil
import envars

_last_activity_time = time.time()

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
    if envars.AUTO_RUN():
        sio.emit('frame_request')


@sio.event
def disconnect():
    print('disconnected from server')


def poll_timer():
    global _last_activity_time
    while True:
        time.sleep(envars.POLL_TIME())
        elapsed_time = time.time() - _last_activity_time
        if envars.AUTO_RUN() and elapsed_time > envars.INACTIVITY_THRESHOLD():
            print('No activity. Querying controller again')
            sio.emit('frame_request')


if __name__ == '__main__':
    model = Model()
    sio.connect('http://{}'.format(envars.CONTROLLER_ADDRESS()))
    poll_timer()
