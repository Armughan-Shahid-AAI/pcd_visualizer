import json

import numpy as np
import os, re
from os import path
import sys
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from matplotlib.colors import to_rgba


def read_pickle(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# Get key for sorting
def int_key(input_str):
    output = re.findall(r".*?([0-9]+.*?).*", input_str)
    if len(output) == 0:
        return None
    else:
        return int(output[0])


# Change colour to rgba
def to_rgba_int(colour_int):
    rgba_float = to_rgba(colour_int)
    return (int(rgba_float[0] * 255),
            int(rgba_float[1] * 255),
            int(rgba_float[2] * 255),
            int(rgba_float[3] * 255))


# Get colour for bounding box
def get_colour(colour, rgb=False):
    """
    param:
    colour_ids list of integer values
    """
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    max_colours = len(cycle)

    if type(colour) is list:
        return [cycle[int(i) % max_colours] if not rgb else to_rgba_int(cycle[int(i) % max_colours])
                for i in colour]
    else:
        return cycle[int(colour) % max_colours] if not rgb else to_rgba_int(cycle[int(colour) % max_colours])


# Scan data directory, sort and return list of images.
def get_images_list(data_dir, sort=True):
    """
    data_dir; path to the folder
    """
    if sort:
        image_names = [data_dir + '/' + file_name for file_name in
                       sorted(os.listdir(data_dir), key=int_key) if file_name is not None]
    else:
        image_names = [data_dir + '/' + file_name for file_name in sorted(os.listdir(data_dir))]

    return image_names


def get_bbox_xy(frame_idx, actor_idx, actors_data_pd):
    current_frame_bboxes = actors_data_pd.iloc[frame_idx]

    actor_id = current_frame_bboxes.bbox_id[actor_idx]
    actor_bbox = current_frame_bboxes.b_boxes[actor_idx]

    return actor_id, actor_bbox


def load_images(path=None):
    return sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]), reverse=False)


def draw_on_image(image,
                  bbox_id,
                  lane_id,
                  ego_lane_id,
                  bbox,
                  actor_distance,
                  detection_size=(416, 416)):
    # font_name = './FiraMono-Medium.otf'
    # font = ImageFont.truetype(font=font_name, size=np.floor(2e-2 * image.size[1]).astype('int32') + 5)
    draw = ImageDraw.Draw(image)

    bbox = np.array(bbox)
    color = get_colour(bbox_id, rgb=True)

    bbox_text = "%d  %d" % (bbox_id, lane_id)

    text_size = draw.textsize(bbox_text)
    detection_size, original_size = np.array(detection_size), np.array(image.size)
    ratio = original_size / detection_size
    bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

    draw.rectangle(bbox, outline=color)

    text_origin = bbox[:2] - np.array([0, text_size[1]])
    draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], fill=color)
    draw.text(tuple(text_origin), bbox_text, fill=(0, 0, 0))

    ego_text = "%d" % (ego_lane_id)
    # font = ImageFont.truetype(font=font_name, size=50)
    text_origin = (image.size[0] / 2, image.size[1] - 100)

    # Distance
    bbox_text = "{}m".format(np.round(actor_distance, 1))
    # font = ImageFont.truetype(font=font_name, size=np.floor(0.3 * (bbox[2] - bbox[0])).astype('int32'))
    text_size = draw.textsize(bbox_text)
    text_origin = (bbox[2] - (bbox[2] - bbox[0]) / 2 - text_size[0] / 2, bbox[3] - (bbox[3] - bbox[1]))

    return image


def create_and_save_image(frame_idx, offset=0):
    current_frame_xy = actors_data_pd.iloc[frame_idx]

    im_path = images_list[frame_idx + offset]

    path = "/home/dane/Work/AUDI/Data/cam_front_center_filt/"

    image = Image.open(os.path.join(path, im_path))

    for actor_idx in range(len(current_frame_xy.bbox_id)):

        actor_id, actor_bbox = get_bbox_xy(frame_idx, actor_idx, actors_data_pd)
        if actor_bbox is not None:
            image = draw_on_image(image, actor_id, 0, 0, actor_bbox[:4],
                                  0)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.show()
    plt.axis("off")
    plt.close("off")


def plot_actor_bbox(actors_data_pd, image, frame_idx, offset=0):

    current_frame_xy = actors_data_pd.iloc[frame_idx]

    for actor_idx in range(len(current_frame_xy.bbox_id)):

        actor_id, actor_bbox = get_bbox_xy(frame_idx, actor_idx, actors_data_pd)

        if actor_bbox is not None:
            image = draw_on_image(image, actor_id, 0, 0, actor_bbox[:4],
                                  0)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
    # plt.close("off")

# --------- Load images -----------------
# root_path = "PATH/TO/ai.aai.scenario.cloning"
#
# # Final stage tracking results.
# trajectory_dir = os.path.join(
#     root_path, "/home/dane/Work/AUDI/Data/actor_tracking_results_road1/road1_results/trajectory_extraction/tracking"
#                "/final_stage")
#
# # Bounding boxes intermediate results file path.
# trajectory_file_path = path.join(trajectory_dir, "real_frames_bboxes_filtered_tracked_joined.pkl")
#
#
# imgs_path = "/home/dane/Work/AUDI/Data/cam_front_center_filt/"
#
# with open(trajectory_file_path, 'rb') as f:
#     actors_data_pd = pickle.load(f)
#
# # test
# print(actors_data_pd.columns)
# print(len(actors_data_pd.iloc[117].b_boxes))
#
# images_list = load_images(imgs_path)
#
# create_and_save_image(116)