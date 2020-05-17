import onnxruntime as rt
import numpy as np
from PIL import Image
import envars


class OnnxModel():
    def __init__(self, model_path='human-pose-estimation.onnx'):
        self.name = model_path.split('.')[0]
        self.sess = rt.InferenceSession(model_path)
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()

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
        # Resize to 256 in height dimension
        orig_width, orig_height = pil_image.size
        scale = orig_height / 256
        image = pil_image.resize((int(orig_width/scale),
                                 int(orig_height/scale)), Image.BILINEAR)
        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
        print(image.shape)
        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])
        # Pad width dimension
        padded_image = np.zeros((3, 256, 456), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image
        # Normalize
        mean_vec = np.array([128, 128, 128])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]
        # add batch dimension
        image = np.expand_dims(image, axis=0)
        # return in onnx tensor format
        return {'data': image}

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
        annotations = []
        return {'name': self.name, 'annotations': annotations}


if __name__ == '__main__':
    imarray = np.random.rand(1920, 1080, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    model = OnnxModel()
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    annotations = model.postprocess(img, output_dict)
    print(annotations)
