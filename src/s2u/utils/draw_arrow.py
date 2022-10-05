import open3d as o3d
from torch import Tensor
import numpy as np
import pickle

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
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat
 
def get_arrow(begin=[0,0,0],vec=[0,0,1],color = [0,1,0]):
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    end = np.add(begin,vec)
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])
 
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1 ,
        cone_radius=0.06 * 1,
        cylinder_height=0.8 * 1,
        cylinder_radius=0.04 * 1
    )
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.compute_vertex_normals()
    
    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow


def get_axis(pc, screw_axis, screw_moment):
    bound_max = pc.max(0)
    bound_min = pc.min(0)

    screw_point = np.cross(screw_axis, screw_moment)
    t_min = (bound_min - screw_point) / screw_axis
    t_max = (bound_max - screw_point) / screw_axis
    axis_index = np.argmin(np.abs(t_max - t_min))
    start_point = screw_point + screw_axis * t_min[axis_index]
    end_point = screw_point + screw_axis * t_max[axis_index]

    return start_point, end_point

def draw_pointcloud(points:np.ndarray, start_point:np.ndarray = [], end_point:np.ndarray = [], start_point_gt = [], end_point_gt = [], colors = []):
    geo_list = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not colors == []:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geo_list.append(pcd)
    if not len(start_point)==0:
        mesh_arrow = get_arrow(begin=start_point, vec=end_point,color = [1,0,0])
        geo_list.append(mesh_arrow)
    if not len(start_point_gt)==0:
        mesh_arrow_2 = get_arrow(begin=start_point_gt, vec=end_point_gt, color = [0,1,0])
        geo_list.append(mesh_arrow_2)
    o3d.visualization.draw_geometries(geo_list)

def save_pointcloud(points:np.ndarray, start_point:np.ndarray = [], end_point:np.ndarray = [], save_dir = ''):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_arrow = get_arrow(begin=start_point, vec=end_point)
    o3d.io.write_triangle_mesh(filename = save_dir, mesh = mesh_arrow)

if __name__=='__main__':
    pointcloud = np.load("/home/douge/Datasets/Motion_Dataset_v0/pointcloud/windmill/windmill_0000_000080108_pc.npy")
    draw_pointcloud(pointcloud[:,:3])