import socketio
import time
import envars

_last_activity_time = time.time()

sio = socketio.Client()

_counter = 0
_model_name = 'box_test'

@sio.event
def connect():
    print('connection established')
    sio.emit('register', _model_name)
    sio.emit('frame_request')


@sio.on("process_frame")
def process_frame(data_url):
    global _last_activity_time
    global _counter
    start_time = time.time()
    _last_activity_time = start_time
    # build annotation
    payload = {'name': _model_name,
               'annotations': [
                   {'kind': 'box',
                    'x': _counter,
                    'y': 0,
                    'width': 50,
                    'height': 50,
                    'label': 'box_test',
                    'confidence': 1}]}
    end_time = time.time()
    _last_activity_time = end_time
    print("[INFO] {} processing time: {:.6f} seconds"
          .format(_model_name, end_time - start_time))
    sio.emit('frame_complete', payload)
    _counter = (_counter + 10) % 400
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
    sio.connect('http://{}'.format(envars.CONTROLLER_ADDRESS()))
    poll_timer()
