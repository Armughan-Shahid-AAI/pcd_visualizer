import csv
import glob
import json
import os
import pickle
import time
import operator
from functools import reduce
from os import path

import cv2

import open3d as o3
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np

from pathlib import Path
from collections import namedtuple, OrderedDict

from PIL import Image
# from pyntcloud import PyntCloud

from cov import get_principal_axes_bbox2d, get_principal_axes_bbox3d
from detections import plot_actor_bbox, get_bbox_xy
from sensors import LidarSensor, CameraSensor
from transform import project_lidar_from_to
from vis import create_open3d_pc, vis_projected_2d, vis_project_bev_bbox, vis_o3_pcl, vis_projected2d

Point3D = namedtuple('Point3D', 'y z x u v r d')


def load_sensor_setup(config_file_path: str):
    cameras, lidars = dict(), dict()
    with open(config_file_path, 'r') as f:
        cfg = json.load(f)

    for sensor in cfg['lidars']:
        lidars[sensor] = LidarSensor(sensor, cfg['lidars'][sensor]['view'])

    for sensor in cfg['cameras']:
        cameras[sensor] = \
            CameraSensor(sensor,
                         cfg['cameras'][sensor]['view'],
                         cfg['cameras'][sensor]['tstamp_delay'],
                         cfg['cameras'][sensor]['Lens'],
                         cfg['cameras'][sensor]['CamMatrix'],
                         cfg['cameras'][sensor]['CamMatrixOriginal'],
                         cfg['cameras'][sensor]['Distortion'],
                         cfg['cameras'][sensor]['Resolution'])

    return lidars, cameras


def load_lidar_csv_frame(file_path: str) -> dict:
    s = time.perf_counter()
    df = pd.read_csv(file_path)
    print("Loading .csv point cloud into pandas: {0} ms.", time.perf_counter() - s)

    return {
        'points': df[['X', 'Y', 'Z']].values,
        'reflectance': df['intensity'].values,
        'timestamp': df['timestamp'].values,
        'distance': df['distance_m'].values,
        'lidar_id': df['laser_id'].values
    }


def filter_pcl(lidar_pcl: dict, axis=0, filter_val=0.0):
    # TODO: default focal length filter val
    return {key: val[lidar_pcl['points'][:, axis] > filter_val] for key, val in lidar_pcl.items()}


def filter_pcl_generic(lidar_pcl: dict, datatype: str = 'points', axis=0, filter_val=0.0):
    # TODO: default focal length filter val
    return {key: val[lidar_pcl[str(datatype)][:, axis] > filter_val] for key, val in lidar_pcl.items()}


def project_to_cam_frame(points3d, camera_sensor: CameraSensor, use_distortion: int = 0):
    projected_pts, jacobian = cv2.projectPoints(
        points3d.T,
        (0, 0, 0), (0, 0, 0),
        np.asarray(camera_sensor.cam_matrix_original),
        np.asarray(camera_sensor.distortion) * use_distortion,
    )
    return projected_pts.reshape(-1, 2)


def change_axes(lidar_points):
    return [-lidar_points[:, 1], -lidar_points[:, 2], lidar_points[:, 0]]


def select_points_multiple_bboxes():
    # select and select .. prior to filter
    raise NotImplementedError


def select_points(lidar_pcl, bbox_p1, bbox_p2):
    select = [bbox_p1[0] < pt[0] < bbox_p2[0] and
              bbox_p1[1] < pt[1] < bbox_p2[1] for pt in lidar_pcl['points_uv'][:, :]]

    return {key: val[select] for key, val in lidar_pcl.items()}


# def create_voxelgrid(points_in, size=16):
#     d = {
#         'x': [p.x for p in points_in],
#         'y': [p.y for p in points_in],
#         'z': [p.z for p in points_in],
#     }
#     cloud = PyntCloud(pd.DataFrame(d))
#
#     # cloud.plot(mesh=False, backend='matplotlib')
#     voxelgrid_id = cloud.add_structure("voxelgrid", n_x=size, n_y=size, n_z=size)
#
#     # new_cloud = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
#     voxelgrid = cloud.structures[voxelgrid_id]
#
#     voxelgrid.plot(d=3, mode="density", cmap="hsv")
#     return voxelgrid


def load_image(path):
    return Image.open(path)
    # return plt.imread(path)


def calc_distance(lidar_pcl, method='closest'):

    if len(lidar_pcl['points_uv']) < 1:
        return None
    if method == 'closest':
        return min(lidar_pcl['distance'])
    elif method == 'average':
        return np.mean(lidar_pcl['distance'])
    elif method == 'pcbbox':
        ptsx = lidar_pcl['points'][:, 0]
        ptsy = lidar_pcl['points'][:, 1]
        p1, p2 = get_principal_axes_bbox2d(x=ptsx, y=ptsy)
        # TODO: bbox too few points (handle different cases)
        return
    else:
        raise NotImplementedError(method)


def load_images(path=None):
    return sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]), reverse=False)


def convert_bbox(bbox, image):
    detection_size=(416, 416)
    bbox = np.array(bbox)
    detection_size, original_size = np.array(detection_size), np.array(image.size)
    ratio = original_size / detection_size
    bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))
    return bbox


def get_bbox(current_frame_data, actor_idx):
    actor_id = current_frame_data.bbox_id[actor_idx]
    actor_bbox = current_frame_data.b_boxes[actor_idx]
    return actor_id, actor_bbox


def process_frame(img_idx, lidar_frame_path, image_frame_path, front_lidar, front_camera, cuboids = None, show_fig=True):
    """

    :param img_idx:
    :param lidar_frame_path:
    :param image_frame_path:
    :param front_lidar:
    :param front_camera:
    :param bboxes: shape (N,8,3)
    :return:
    """
    print("Processing: ", lidar_frame_path, image_frame_path)
    # 1. Load
    lidar_data = load_lidar_csv_frame(file_path=lidar_frame_path)
    image = load_image(image_frame_path)

    # print("\n\n############points###########")
    # raw_points = lidar_data['points']
    # print (np.min(raw_points,axis=0))
    # print (np.max(raw_points,axis=0))
    # print (raw_points[np.random.randint(low=0, high=len(raw_points), size=20)])
    # print("########points#######\n\n")

    T_lidar_data = project_lidar_from_to(
        lidar_data,
        src_view=front_lidar.view,
        target_view=front_camera.view,
    )

    # 3. filter
    # T_lidar_data = filter_pcl(T_lidar_data, axis=0, filter_val=0)

    # 4. align
    points_3d = np.array(change_axes(T_lidar_data['points']))
    # print ("lidar points ",points_3d.shape, lidar_data['points'].shape, T_lidar_data['points'].shape)
    # 5. project
    projected_points = project_to_cam_frame(points_3d, front_camera)
    T_lidar_data['points_uv'] = projected_points



    if cuboids is not None:
        cuboids_data = {'points': cuboids.reshape(-1, 3)}
        T_cuboids_data = project_lidar_from_to(
            cuboids_data,
            src_view=front_lidar.view,
            target_view=front_camera.view,
        )

        # 3. filter
        # T_cuboids_data = filter_pcl(T_cuboids_data, axis=0, filter_val=0)

        # 4. align
        points_3d = np.array(change_axes(T_cuboids_data['points']))
        # 5. project
        cuboids = project_to_cam_frame(points_3d, front_camera).reshape(-1, 8, 2)
        # print ("cuboids",cuboids)
    # 6. visualize
    fig = vis_projected2d(image, T_lidar_data, cuboids=cuboids, show_fig=show_fig)
    return fig



def get_images_and_lidar_files(lidar_dir, img_dir):
    lidar_csv_path = lidar_dir / '*.csv'
    lidar_files = sorted(glob.glob(str(lidar_csv_path)))
    image_files = load_images(img_dir)  # already sorted
    image_files = [os.path.join(img_dir, f) for f in image_files]

    return lidar_files, image_files


def load_sce_results():
    root_path = "/home/dane/Work/AUDI/Data/actor_tracking_results_road1/road1_results/trajectory_extraction/"

    # Final stage tracking results.
    trajectory_dir = os.path.join(
        root_path, "tracking/final_stage")

    # Bounding boxes intermediate results file path.
    trajectory_file_path = path.join(trajectory_dir, "real_frames_bboxes_filtered_tracked_joined.pkl")
    actor_distances_path = path.join(root_path, "ActorTrajectories/actor_absolute_trajectories.pkl")

    with open(trajectory_file_path, 'rb') as f:
        actors_data_pd = pickle.load(f)

    with open(actor_distances_path, 'rb') as f:
        actors_distances_pd = pickle.load(f)

    return actors_data_pd, actors_distances_pd


