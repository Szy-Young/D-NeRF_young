import numpy as np
import open3d as o3d
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from utils.point_util import fps_downsample_np


COLOR20 = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])


def build_colored_pointcloud(pc, color):
    """
    :param pc: (N, 3).
    :param color: (N, 3).
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud


lines = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
box_colors = [[0, 1, 0] for _ in range(len(lines))]

def build_bbox3d(boxes):
    """
    :param boxes: List [(8, 3), ...].
    """
    line_sets = []
    for corner_box in boxes:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corner_box)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(box_colors)
        line_sets.append(line_set)
    return line_sets

def bound_to_box(bounds):
    """
    :param bounds: List [(3, 2), ...].
    """
    boxes = []
    for bound in bounds:
        corner_box = np.array([[bound[0, 0], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 0]],
                               [bound[0, 1], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 0], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 0]],
                               [bound[0, 1], bound[1, 1], bound[2, 1]],
                               [bound[0, 0], bound[1, 1], bound[2, 1]]])
        boxes.append(corner_box)
    return boxes


# def build_point_flow(pc, flow, with_point=False, n_sample_point=None):
#     """
#     :param pc: (N, 3).
#     :param flow: (N, 3).
#     """
#     # Downsample
#     if n_sample_point is not None:
#         fps_idx = fps_downsample_np(pc, n_sample_point=n_sample_point)
#         pc, flow = pc[fps_idx], flow[fps_idx]
#
#     n_point = pc.shape[0]
#     pc_2 = np.concatenate([pc, pc + flow], 0)
#     line_idx = [[n, n+n_point] for n in range(n_point)]
#     line_idx = np.array(line_idx, dtype=np.int32)
#     pcd_flow = o3d.geometry.LineSet()
#     pcd_flow.points = o3d.utility.Vector3dVector(pc_2)
#     pcd_flow.lines = o3d.utility.Vector2iVector(line_idx)
#     ret_list = [pcd_flow]
#
#     if with_point:
#         pcd_1 = o3d.geometry.PointCloud()
#         pcd_1.points = o3d.utility.Vector3dVector(pc)
#         color1 = np.tile(COLOR20[0] / 255., [n_point, 1])
#         pcd_1.colors = o3d.utility.Vector3dVector(color1)
#         ret_list.append(pcd_1)
#         # pcd_2 = o3d.geometry.PointCloud()
#         # pcd_2.points = o3d.utility.Vector3dVector(pc + flow)
#         # color2 = np.tile(COLOR20[1] / 255., [n_point, 1])
#         # pcd_2.colors = o3d.utility.Vector3dVector(color2)
#         # ret_list.append(pcd_2)
#     return ret_list


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat


def pc_flow_to_sphere(pc, flow, radius=0.001, resolution=10, pc_color=None, flow_color=[127, 127, 127],
                      with_point=False, n_sample_point=None):
    """
    Visualize scene flow vectors as arrows.
    :param pc: (N, 3)
    :param flow: (N, 3)
    """
    flow_color = np.array(flow_color)

    # Downsample
    if n_sample_point is not None:
        fps_idx = fps_downsample_np(pc, n_sample_point=n_sample_point)
        pc, flow = pc[fps_idx], flow[fps_idx]
        if pc_color is not None:
            pc_color = pc_color[fps_idx]

    n_point = pc.shape[0]
    meshes = []
    for pid in range(n_point):
        point, point_flow = pc[pid], flow[pid]
        flow_len = np.linalg.norm(point_flow)
        point_flow = point_flow / flow_len
        # flow_len = np.linalg.norm(point_flow) * 10
        mesh = o3d.geometry.TriangleMesh.create_arrow(
            cone_height= 0.2 * flow_len,
            cone_radius= 1.5 * radius,
            cylinder_height= 0.8 * flow_len,
            cylinder_radius= radius,
            resolution=resolution
        )
        mesh.paint_uniform_color(flow_color / 255.)
        rot_mat = caculate_align_mat(point_flow)
        mesh.rotate(rot_mat, center=(0, 0, 0))
        mesh.translate(point)
        meshes.append(mesh)

        if with_point:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=5*radius, resolution=resolution)
            if pc_color is not None:
                mesh.paint_uniform_color(pc_color[pid])
            mesh.translate(point)
            meshes.append(mesh)

    # Merge
    mesh = meshes[0]
    for i in range(1, len(meshes)):
        mesh += meshes[i]
    return mesh


def visualize_point_flow_plt(pc, flow, save_file=None):
    """
    :param pc: (N, 3).
    :param flow: (N, 3).
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]
    flow_x, flow_y, flow_z = flow[:, 0], flow[:, 1], flow[:, 2]
    ax.scatter(pc_x, pc_y, pc_z, c='r', s=1.0)
    ax.quiver(pc_x, pc_y, pc_z, flow_x, flow_y, flow_z)
    plt.show()

    if save_file is not None:
        fig.savefig(save_file)