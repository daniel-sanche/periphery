from PIL import Image
import envars
import argparse
import cv2
import numpy as np
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from image_functions import np_img_to_data_url



class Model():
    def __init__(self, cpu=True, remember_previous=True):
        self.name = 'OpenPose'
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load('./checkpoint_iter_370000.pth', map_location='cpu')
        load_state(net, checkpoint)
        self.net = net.eval()
        if not cpu:
            self.net = self.net.cuda()
        self.cpu = cpu
        self.stride = 8
        self.upsample_ratio = 4
        self.height_size = 256
        self.kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
        self.connections = [
        ('nose', 'r_eye'), ('r_eye', 'r_ear'),
        ('nose', 'l_eye'), ('l_eye', 'l_ear'),
        ('nose', 'neck'),
        ('neck', 'r_sho'), ('r_sho', 'r_elb'),('r_elb', 'r_wri'),
        ('neck', 'l_sho'), ('l_sho', 'l_elb'),('l_elb', 'l_wri'),
        ('neck', 'r_hip'),('r_hip', 'r_knee'),('r_knee', 'r_ank'),
        ('neck', 'l_hip'), ('l_hip', 'l_knee'),('l_knee', 'l_ank')]


    def preprocess(self, pil_image):
        """
        Reformat generic PIL image to onnx input form
        """
        image = pil_image
        image = np.array(image).astype(np.uint8)

        # img2 = cv2.imread('data/preview.jpg', cv2.IMREAD_COLOR)

        # print(image.shape, image[0,0,0])
        # print(img2.shape, img2[0,0,0])
        return image

    def run(self, img):
        """
        Run one prediction
        """
        heatmaps, pafs, scale, pad = infer_fast(self.net, img,
                self.height_size, self.stride, self.upsample_ratio, self.cpu)
        return {'heatmaps':heatmaps, 'pafs':pafs, 'scale':scale, 'pad':pad}

    def postprocess(self, orig_image, output_dict):
        """
        Reformat output into standard annotations
        """
        heatmaps = output_dict['heatmaps']
        pafs = output_dict['pafs']
        scale = output_dict['scale']
        pad = output_dict['pad']
        num_keypoints = Pose.num_kpts

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
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

        annotations = []
        for pose_idx, pose in enumerate(current_poses):
            person_label = 'person_{}'.format(pose_idx)
            if envars.OUTPUT_POSES():
                point_arr = []
                found_points = set([])
                for idx in range(pose.keypoints.shape[0]):
                    x = int(pose.keypoints[idx, 0])
                    y = int(pose.keypoints[idx, 1])
                    label = self.kpt_names[idx]
                    if x >= 0 and y >= 0:
                        point_arr.append({'x': x, 'y':y, 'label':label})
                        found_points.add(label)
                links = [{'from': x[0], 'to': x[1]}
                         for x in self.connections
                         if x[0] in found_points and x[1] in found_points]
                annotations.append({'kind':'lines',
                                    'label': person_label,
                                    'confidence': pose.confidence,
                                    'points': point_arr,
                                    'links':links})
            if envars.OUTPUT_BOXES():
                bbox_annotation = {'kind':'box',
                                   'x':pose.bbox[0],
                                   'y':pose.bbox[1],
                                   'height':pose.bbox[3],
                                   'width':pose.bbox[2],
                                   'label':person_label,
                                   'confidence':pose.confidence}
                annotations.append(bbox_annotation)


        return {'name': self.name, 'annotations': annotations}


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

if __name__ == '__main__':
    imarray = np.random.rand(1920, 1080, 3) * 255
    img = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    model = Model()
    input_dict = model.preprocess(img)
    output_dict = model.run(input_dict)
    annotations = model.postprocess(img, output_dict)
    print(annotations)
