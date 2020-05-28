import onnxruntime as rt
import numpy as np
from PIL import Image
import envars


class OnnxModel():
    def __init__(self, model_path='tiny-yolov3-11.onnx',
                 class_path='coco_classes.txt'):
        self.name = 'tiny_yolov3'
        self.sess = rt.InferenceSession(model_path)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()
        self.classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

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

    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def preprocess(self, img):
        """
        Reformat generic PIL image to onnx input form
        """
        orig_size = np.array([img.size[1],
                              img.size[0]], dtype=np.float32).reshape(1, 2)
        model_image_size = (416, 416)
        boxed_image = self.letterbox_image(img,
                                           tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return {'input_1': image_data, 'image_shape': orig_size}

    def run(self, input_dict):
        """
        Run one prediction
        """
        pred = self.sess.run(None, input_dict)
        return pred

    def postprocess(self, orig_image, output_dict):
        """
        Reformat output into standard annotations
        """

        boxes = output_dict[0].squeeze()
        scores = output_dict[1].squeeze()
        indices = output_dict[2]

        num_boxes = indices.shape[1]

        print(scores.shape)

        annotations = []
        for i in range(num_boxes):
            class_idx = indices[0, i, 1]
            box_idx = indices[0, i, 2]
            box = boxes[box_idx, :]
            label = self.classes[class_idx]
            score = scores[class_idx, box_idx]
            if score > envars.CONFIDENCE_THRESHOLD():
                x = int(box[1])
                y = int(box[0])
                height = int(box[2]) - y
                width = int(box[3]) - x
                annotation = {'kind': 'box', 'x': x, 'y': y,
                              'width': width, 'height': height,
                              'label': label, 'confidence': float(score)}
                annotations.append(annotation)
        return {'name': self.name, 'annotations': annotations}


if __name__ == '__main__':
    imarray = np.random.rand(1920, 1080, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    model = OnnxModel()
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    annotations = model.postprocess(img, output_dict)
    print(annotations)
