import os
import sys

import numpy as np
import argparse

from run import load_sensor_setup, process_frame

def create_dirs_if_not_exists(directories):
    if type(directories)==str:
        directories=[directories]

    for d in directories:
        if not os.path.isdir(d):
            os.makedirs(d)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--point_cloud_file', help='specify the csv file containing point cloud info')
    parser.add_argument('--img_file', help='specify the img file')
    parser.add_argument('--config_file', help='specify the config json file')
    parser.add_argument('--predictions_file', help='specify the npy predictions file file', default="")
    parser.add_argument('--visualize', action='store_true', help='whether to visualize')
    parser.add_argument('--output_dir', default="predictions_dir", help='whether to visualize')
    parser.add_argument('--lidar_type', default="front_center", help='description of lidar sensor')
    parser.add_argument('--camera_type', default="front_center", help='description of camera sensor')

    args = parser.parse_args(argv)
    return args

def main(args):

    lidar_dict, camera_dict = load_sensor_setup(config_file_path=args.config_file)
    create_dirs_if_not_exists([args.output_dir])
    front_lidar = lidar_dict[args.lidar_type]
    front_camera = camera_dict[args.camera_type]
    detected_cuboids = None if args.predictions_file == "" else np.load(pred_3d_boxes_file).item()
    fig = process_frame(0, args.point_cloud_file, args.img_file, front_lidar, front_camera, show_fig=args.visualize)
    output_path = os.path.splitext(os.path.basename(args.point_cloud_file))[0]
    output_path += "_cam_type_" + args.camera_type + "_lidar_type_" + args.lidar_type + ".png"
    output_path = os.path.join(args.output_dir, output_path)
    fig.savefig(output_path)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)