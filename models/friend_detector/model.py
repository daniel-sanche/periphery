import onnxruntime as rt
import numpy as np
from PIL import Image
import math
import cv2
import envars


class OnnxModel():
    def __init__(self, model_path='updated_arcface.onnx'):
        self.name = 'FriendDetector'
        self.sess = rt.InferenceSession(model_path)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()

        self.threshold = 20
        self.box_scaler = 1.25

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
        confidence_kept = []
        for i, (x, y, w, h) in enumerate(faces):
            c = confidence[i]
            print(c)
            if c >= self.threshold:
                h_p = int(h * self.box_scaler)
                w_p = int(w * self.box_scaler)
                crop_img = img[y:y+h_p, x:x+w_p,:]
                crop_img = cv2.resize(crop_img, (112, 112))
                crop_img = np.transpose(crop_img, [2, 0, 1])
                crop_img = np.expand_dims(crop_img, axis=0)
                cropped_list.append(crop_img)
                faces_kept.append((int(x), int(y), w_p, h_p))
                confidence_kept.append(float(c))
        # return in onnx tensor format
        return {'images': cropped_list, 'boxes': faces_kept, 'confidence':confidence_kept}

    def run(self, input_dict):
        """
        Run one prediction
        """
        image_list = input_dict.get('images')
        vector_list = []
        for img in image_list:
            vector_list.append(self.sess.run(None, {'data':img})[0])
        return {'boxes': input_dict.get('boxes'),
                'confidence': input_dict.get('confidence'),
                'vectors': vector_list}

    def postprocess(self, orig_image, output_dict):
        """
        Reformat output into standard annotations
        """
        boxes = output_dict['boxes']
        vectors = output_dict['vectors']
        confidences = output_dict['confidence']
        num_faces = len(boxes)

        annotations = []
        for i in range(num_faces):
            (x, y, w, h) = boxes[i]
            vector = np.squeeze(vectors[i])
            confidence = confidences[i]
            print(confidence)
            annotations.append(
                {'kind': 'box', 'x': x, 'y': y, 'width': w, 'height': h,
                'label': 'face', 'confidence': confidence})

        return {'name': self.name, 'annotations': annotations}

if __name__ == '__main__':
    imarray = np.random.rand(1920, 1080, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    model = OnnxModel()
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    annotations = model.postprocess(img, output_dict)
    print(annotations)
