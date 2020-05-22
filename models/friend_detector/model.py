import onnxruntime as rt
import numpy as np
from PIL import Image
import math
import cv2
import envars
import os
from sklearn import preprocessing
from scipy.special import softmax
import train
import sys

class OnnxModel():
    def __init__(self, model_path='updated_arcface.onnx',
            images_path='./dataset', pickle_path='./data.pickle'):
        self.name = 'FriendDetector'
        self.sess = rt.InferenceSession(model_path)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()

        # load dataset
        image_label_data = {}
        pickle_label_data = {}
        if envars.USE_IMAGE_DATASET():
            print('loading from images in: {}'.format(images_path))
            image_label_data = train.images_to_dict(images_path, self)
        if envars.USE_PICKLE_DATASET() and os.path.exists(pickle_path):
            print('loading from pickle: {}'.format(pickle_path))
            pickle_label_data = train.load_dict_from_disk(pickle_path)
        combined_labels = train.merge_dicts(image_label_data, pickle_label_data)
        if envars.SAVE_DATASET_TO_PICKLE():
            print('saving dataset to: {}'.format(pickle_path))
            train.save_dict_to_disk(combined_labels, pickle_path)
        X, y = train.dict_to_compressed_mat(combined_labels)
        if not y:
            print('data not found. aborting')
            sys.exit(1)

        self.X = X
        self.labels = y
        print('found labels: {}'.format(y))
        print('dataset shape: {}'.format(X.shape))

        # print input/output details
        print("backend: {}".format(rt.get_device()))
        print("inputs:")
        for i, input in enumerate(self.inputs):
            print("{} - {}: {} - {}".format(
                i, input.name, input.shape, input.type))
        print("outputs:")
        for i, output in enumerate(self.outputs):
            print("{} - {}: {} - {}".format(
                i, output.name, output.shape, output.type))

    def preprocess(self, pil_image):
        """
        Reformat generic PIL image to onnx input form
        """
        img = np.array(pil_image)

        # extract faces
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces, confidence = face_cascade.detectMultiScale2(gray, 1.1, 4)

        cropped_list = []
        img = img.astype(np.float32)
        faces_kept = []
        for i, (x, y, w, h) in enumerate(faces):
            c = confidence[i]
            if c >= envars.DETECTION_CONFIDENCE_THRESHOLD():
                crop_img = img[y:y+h, x:x+w,:]
                crop_img = cv2.resize(crop_img, (112, 112))
                crop_img = np.transpose(crop_img, [2, 0, 1])
                crop_img = np.expand_dims(crop_img, axis=0)
                cropped_list.append(crop_img)
                faces_kept.append((int(x), int(y), int(w), int(h)))
        # return in onnx tensor format
        return {'images': cropped_list, 'boxes': faces_kept}

    def run(self, input_dict):
        """
        Run one prediction
        """
        image_list = input_dict.get('images')
        vector_list = []
        for img in image_list:
            unnormalized = self.sess.run(None, {'data':img})[0]
            vector_list.append(unnormalized)
        return {'boxes': input_dict.get('boxes'),
                'vectors': vector_list}

    def postprocess(self, orig_image, output_dict):
        """
        Reformat output into standard annotations
        """
        boxes = output_dict['boxes']
        vectors = output_dict['vectors']
        num_faces = len(boxes)

        annotations = []
        for i in range(num_faces):
            (x, y, w, h) = boxes[i]
            vector = np.squeeze(vectors[i])
            label, score = self.find_closest(vector)
            # print(confidence)
            annotations.append(
                {'kind': 'box', 'x': x, 'y': y, 'width': w, 'height': h,
                'label': label, 'confidence': score})

        return {'name': self.name, 'annotations': annotations}

    def find_closest(self, vector):
        repeated = np.repeat(vector[np.newaxis, :], self.X.shape[0], axis=0)
        difference = repeated - self.X
        distances = np.linalg.norm(difference, axis=1)
        normalized_scores = softmax(-distances)
        print(distances)
        print(["%.2f" % v for v in normalized_scores])
        print(self.labels)

        idx = np.argmax(normalized_scores)
        score = np.amax(normalized_scores)
        # absolute_score = np.amin(average_distances)
        label = self.labels[idx]

        if score < envars.RECOGNITION_CONFIDENCE_THRESHOLD():
            label = envars.UNKNOWN_LABEL()
        return label, float(score)

if __name__ == '__main__':
    imarray = np.random.rand(1920, 1080, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    model = OnnxModel()
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    annotations = model.postprocess(img, output_dict)
    print(annotations)
