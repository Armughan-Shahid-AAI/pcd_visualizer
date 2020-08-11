import open3d as o3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

cuboid_vertex_indexes = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7]
]

def vis_projected2d(img, lidar_pcl, bbox=None, distance='Nan', cuboids=None, show_fig=True):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    points_uv = lidar_pcl['points_uv']
    reflectance = lidar_pcl['reflectance']
    h, w, *_ = img.shape

    pts, rs = list(zip(*[(pt, r) for pt, r in zip(points_uv, reflectance) if 0 <= pt[0] < w and 0 <= pt[1] < h]))
    pts_xy = list(zip(*pts))

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.scatter(pts_xy[0], pts_xy[1], c=rs, s=2.2, cmap='hot')
    if bbox is not None:
        p1, p2 = bbox
        rect = Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], linewidth=1, edgecolor='r', facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)

        plt.text(p1[0], p1[1] - (p1[1] - p2[1]) + 30, str(distance) + " m.", c='g', fontsize=24)

    if cuboids is not None:
        for cube in cuboids:
            for vertex_index in cuboid_vertex_indexes:
                p1, p2 = cube[vertex_index[0]], cube[vertex_index[1]]
                rect = Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], linewidth=1, edgecolor='r', facecolor='none')
                ax = plt.gca()
                ax.add_patch(rect)

                circ1 = Circle(p1, 20, color='b', alpha=0.5)
                circ2 = Circle(p2, 20, color='b', alpha=0.5)
                ax.add_patch(circ1)
                ax.add_patch(circ2)



    plt.title("Projected lidar points.")
    plt.axis('off')
    if show_fig:
        plt.show()
    return fig


def vis_o3_pcl(points, downsample=False, voxel_size=0.3):
    pcl_in = {
        'points': [[p.x, p.y, p.z] for p in points],
        'reflectance': [p.r for p in points]
    }

    pcd = create_open3d_pc(pcl_in)
    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    o3.visualization.draw_geometries([pcd])


def vis_project_bev_bbox(lidar_pts, points_in):
    plt.figure(figsize=(15, 15))
    plt.scatter([p.y for p in points_in], [p.x for p in points_in], s=0.25)
    plt.scatter([p[0] for p in lidar_pts],
                [p[2] for p in lidar_pts], s=0.015)

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.show()


def vis_project_bev(lidar_pts):
    plt.figure(figsize=(15, 15))
    plt.scatter([p[1] for p in lidar_pts], [p[0] for p in lidar_pts], s=0.015)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)


def vis_project_2d_inv(image, projected, points_in):
    plt.imshow(image)
    print()
    plt.scatter([1920 - p[0] for p in projected], [1208 - p[1] for p in projected],
                c=[p[2] for p in projected],
                s=1)
    plt.scatter([1920 - p.u for p in points_in], [1208 - p.v for p in points_in], s=1)
    plt.xlim(0, 1920)
    plt.ylim(1208, 0)
    plt.show()


def vis_projected_2d(image, projected, points_in, p1, p2, min_d):
    plt.figure(figsize=(12,7))
    import scipy
    # plt.imshow(scipy.ndimage.rotate(image, 180), origin='upper')
    # plt.imshow(image, origin='upper')
    # plt.scatter(projected[:, 0], projected[:, 1], s=0.6)
    image = image.rotate(180)
    image = np.array(image)
    plt.imshow(image)

    plt.scatter([p[0] for p in projected], [p[1] for p in projected],
                c= [p[2] for p in projected], cmap='hot',
                s=5.6)
    plt.scatter([p.u for p in points_in], [p.v for p in points_in], c='g', s=3.6)
    ax = plt.gca()

    from matplotlib.patches import Rectangle
    rect = Rectangle(p1, p2[0]-p1[0], p2[1]-p1[1],linewidth=1,edgecolor='r',facecolor='none')

    plt.text(p1[0], p1[1] - (p1[1] - p2[1]) + 30, str(min_d) + " m.", fontsize=24)

    ax.add_patch(rect)
    plt.xlim(0, 1920)
    plt.ylim(0, 1208)
    plt.show()
    return plt


# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)


def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()

    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)

        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0

    pcd.colors = o3.utility.Vector3dVector(colours)

    return pcd