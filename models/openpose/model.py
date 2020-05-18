import onnxruntime as rt
import numpy as np
from PIL import Image
import envars
from keypoints import extract_keypoints, group_keypoints
from pose import Pose, track_poses


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
        scale = 256 / orig_height
        image = pil_image.resize((int(orig_width*scale),
                                 int(orig_height*scale)), Image.BILINEAR)
        print(image.size)
        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
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
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        previous_poses = []
        heatmaps = np.squeeze(output_dict[2])
        pafs = np.squeeze(output_dict[3])
        orig_width, orig_height = orig_image.size
        scale = 256 / orig_height
        pad = [115, 0] # fix me

        # extract keypoints
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        # extract poses
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        # if track:
        #     track_poses(previous_poses, current_poses, smooth=smooth)
        #     previous_poses = current_poses
        for pose in current_poses:
            print(pose)

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
