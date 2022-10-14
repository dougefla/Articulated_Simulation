import trimesh
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from tqdm import tqdm
import numpy as np

from s2u.simulation import ArticulatedObjectManipulationSim
from s2u.utils.axis2transform import axis2transformation
from s2u.utils.saver import get_mesh_pose_dict_from_world
from s2u.utils.visual import as_mesh
from s2u.utils.implicit import sample_iou_points_occ
from s2u.utils.io import write_data
from s2u.utils.draw_arrow import draw_pointcloud

import os

def binary_occ(occ_list, idx):
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    return occ_fore, occ_back


def sample_occ(sim, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    mesh_dict = {}
    whole_scene = trimesh.Scene()
    for k, v in result_dict.items():
        scene = trimesh.Scene()
        for mesh_path, scale, pose in v:
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(scale)
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
        mesh_dict[k] = as_mesh(scene)
    points_occ, occ_list = sample_iou_points_occ(mesh_dict.values(),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list

def sample_occ_binary(sim, mobile_links, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    new_dict = {'0_0': [], '0_1': []}
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    whole_scene = trimesh.Scene()
    static_scene = trimesh.Scene()
    mobile_scene = trimesh.Scene()
    for k, v in result_dict.items():
        body_uid, link_index = k.split('_')
        link_index = int(link_index)
        if link_index in mobile_links:
            new_dict['0_1'] += v
        else:
            new_dict['0_0'] += v
        for mesh_path, scale, pose in v:
            if mesh_path.startswith('#'): # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
            else:
                mesh = trimesh.load(mesh_path)
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
            if link_index in mobile_links:
                mobile_scene.add_geometry(mesh)
            else:
                static_scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
    static_mesh = as_mesh(static_scene)
    mobile_mesh = as_mesh(mobile_scene)
    points_occ, occ_list = sample_iou_points_occ((static_mesh, mobile_mesh),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list, new_dict


def main(args, rank):
    data_root = '/home/douge/Datasets/Motion_Dataset_v0'
    urdf_root= os.path.join(data_root, 'urdf')
    
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    sim = ArticulatedObjectManipulationSim(args.object_set,
                                           size=0.3,
                                           gui=args.sim_gui,
                                           global_scaling=args.global_scaling,
                                           dense_photo=args.dense_photo,
                                           urdf_root=urdf_root)
    scenes_per_worker = args.num_scenes // args.num_proc
    pbar = tqdm(total=scenes_per_worker, disable=rank != 0)
    
    if rank == 0:
        print(f'Number of objects: {len(sim.object_urdfs)}')
    
    for _ in range(scenes_per_worker):
        try:
            sim.reset(canonical=args.canonical)
            object_path = str(sim.object_urdfs[sim.object_idx])
            
            
            result = collect_observations(
                sim, args)
            
            result['object_path'] = object_path
            
            write_data(args.root, args.object_set, result)
            
            pbar.update()
        except:
            print("Skip: ")
            continue
    
    pbar.close()
    print('Process %d finished!' % rank)


def get_limit(v, args):
    joint_type = v[2]
    # specify revolute angle range for shape2motion
    if joint_type == 0 and not args.is_syn:
        if args.pos_rot:
            lower_limit = 0
            range_lim = np.pi / 2
            higher_limit = np.pi / 2
        else:
            lower_limit = - np.pi / 4
            range_lim = np.pi / 2
            higher_limit = np.pi / 4
    else:
        lower_limit = v[8]
        higher_limit = v[9]
        range_lim = higher_limit - lower_limit
    return lower_limit, higher_limit, range_lim

def collect_observations(sim, args):

    joint_info = sim.get_joint_info()
    all_joints = list(joint_info.keys())
    joint_index = all_joints.pop(np.random.randint(len(all_joints)))

    for x in all_joints:
        v = joint_info[x]
        lower_limit, higher_limit, range_lim = get_limit(v, args)
        start_state = np.random.uniform(lower_limit, higher_limit)
        sim.set_joint_state(x, start_state)

    v = joint_info[joint_index]
    axis, moment, point_on_axis = sim.get_joint_screw(joint_index)
    joint_type = v[2]
    
    _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])

    result = {
            'pc_start': start_pc,
            'pc_seg_start': start_seg_label,
            'screw_axis': axis,
            'screw_moment': moment,
            'joint_type': joint_type,
            'joint_index': joint_index,
            'start_mesh_pose_dict': start_mesh_pose_dict,
            'point_on_axis': point_on_axis,
        }

    return result

def multi_func(args, object_set):

    args.object_set = object_set
    if 'syn' in args.object_set:
        args.is_syn = True
    else:
        args.is_syn = False
    
    if not os.path.exists(args.root / "scenes"):
        (args.root / "scenes").mkdir(parents=True)
    if args.num_proc > 1:
        #print(args.num_proc)
        pool = mp.get_context("spawn").Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default="/home/douge/Datasets/Motion_Dataset_v0")
    parser.add_argument("--object-set", type=str)
    parser.add_argument("--num-scenes", type=int, default=10000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--range-scale", type=float, default=0.3)
    parser.add_argument("--num-point-occ", type=int, default=100000)
    parser.add_argument("--occ-var", type=float, default=0.005)
    parser.add_argument("--pos-rot", type=int, required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--sample-method", type=str, default='mix')
    parser.add_argument("--rand-state", action="store_true", help='set static joints at random state')
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")


    args = parser.parse_args()

    # object_list = ["washing_machine", "laptop", "screwdriver", "seesaw", "wine_bottle", "valve", "rocking_chair", "windmill"]
    object_list = ["laptop"]

    for i in object_list:
        multi_func(args,i)


'''
bucket  revolver	kettle	excavator	washing_machine	motorbike	fan	    lamp	laptop	swiss_army_knife	bike	scissors	cannon	screwdriver	seesaw	closestool	water_bottle	wine_bottle	folding_chair	tank	handcart	door	swivel_chair	lighter	cabinet	plane	carton	eyeglasses	refrigerator	watch	faucet	stapler	globe	car	    valve	skateboard	pen     swing	window	oven	rocking_chair	windmill	clock	helicopter	
14      25          61      39          62              107         64      68      86      17                  63      26          102     70          23      64          56              17      	21          	30  	121	        92  	21          	37  	30  	143 	8   	43	        81          	6	    162	    33	    29	    101	    36	    79	        52      36      14  	42      19              78          58	    104	
3/0	    3/1     	1/1 	9/0 	    1/0 	        3/0     	7/0	    5/0 	1/0 	10/1            	5/0	    2/0     	21/0	1/0	        1/0	    2/0     	1/2         	0/1     	2/0	            19/0    6/0     	6/1     1/1             2/1     6/9 	10/0	4/1	    2/0	        2/3	            3/0     4/0	    2/0	    2/0 	17/0    1/0 	8/0         0/2     4/0     2/2	    2/0     1/0             1/0	        3/0     2/0
4	    4	        3	    10	        2	            4	        8	    6	    2	    11	                6	    3	        22	    2	        2	    3	        3	            2	        3	            20	    7	        7	    3	            3	    10	    11	    5	    3	        6	            4	    5	    3	    3	    18	    2	    9	        3	    5	    3	    3	    2	            2           4       3	

'''