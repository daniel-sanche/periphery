From python:3.6

RUN wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

RUN pip install --upgrade cython
RUN pip install numpy>=1.14.0
RUN pip install torch==0.4.1
RUN pip install torchvision>=0.2.1
RUN pip install pycocotools==2.0.0
RUN pip install opencv-python>=3.4.0.14
RUN pip install matplotlib
RUN pip install python-socketio[client]
RUN pip install pillow

# opencv
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev

COPY . .

ENTRYPOINT python socket_client.py
