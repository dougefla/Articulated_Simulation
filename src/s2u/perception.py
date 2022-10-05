from math import cos, sin
import time

import numpy as np
import open3d as o3d

from s2u.utils.transform import Transform


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=color_type,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        extrinsic = extrinsic.as_matrix()
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def integrate_rgb(self, depth_img, rgb, intrinsic, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        extrinsic = extrinsic.as_matrix()
        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        voxels = self._volume.extract_voxel_grid().get_voxels()
        for voxel in voxels:
            i, j, k = voxel.grid_index
            tsdf_grid[0, i, j, k] = voxel.color[0]
        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


def camera_on_sphere(origin, radius, theta, phi) -> Transform:
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/ray_casting.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

from tqdm import tqdm


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.path_npz):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.path_npz))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.path_npz))

        n_files = len(depth_file_names)
        device = o3d.core.Device(config.device)

        voxel_size = 3.0 / 512
        trunc = voxel_size * 4
        res = 8

        if config.integrate_color:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight', 'color'),
                (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
                3.0 / 512, 8, 100000, device)
        else:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight'), (o3c.float32, o3c.float32), ((1), (1)),
                3.0 / 512, 8, 100000, device)

        start = time.time()
        for i in tqdm(range(n_files)):
            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            start = time.time()
            # Get active frustum block coordinates from input
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, config.depth_scale,
                config.depth_max)
            # Activate them in the underlying hash map (may have been inserted)
            vbg.hashmap().activate(frustum_block_coords)

            # Find buf indices in the underlying engine
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            o3d.core.cuda.synchronize()
            end = time.time()

            # Now project them to the depth and find association
            # (3, N) -> (2, N)
            start = time.time()
            extrinsic_dev = extrinsic.to(device, o3c.float32)
            xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                           3:]

            intrinsic_dev = intrinsic.to(device, o3c.float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3c.int64)
            v = (uvd[1] / d).round().to(o3c.int64)
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
                v < depth.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]

            depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
                o3c.float32) / config.depth_scale
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) \
                & (depth_readings < config.depth_max) \
                & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            tsdf[valid_voxel_indices] \
                = (tsdf[valid_voxel_indices] * w +
                   sdf[mask_inlier].reshape(w.shape)) / (wp)
            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                color_readings = color.as_tensor()[v_proj,
                                                   u_proj].to(o3c.float32)

                color = vbg.attribute('color').reshape((-1, 3))
                color[valid_voxel_indices] \
                    = (color[valid_voxel_indices] * w +
                             color_readings[mask_inlier]) / (wp)

            weight[valid_voxel_indices] = wp
            o3d.core.cuda.synchronize()
            end = time.time()

        print('Saving to {}...'.format(config.path_npz))
        vbg.save(config.path_npz)
        print('Saving finished')

    return vbg


if __name__ == '__main__':

    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, jack_jack]',
               default='lounge')
    parser.add('--integrate_color', action='store_true')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config = get_default_dataset(config)

    # Extract RGB-D frames and intrinsic from bag file.
    if config.path_dataset.endswith(".bag"):
        assert os.path.isfile(
            config.path_dataset), (f"File {config.path_dataset} not found.")
        print("Extracting frames from RGBD video file")
        config.path_dataset, config.path_intrinsic, config.depth_scale = extract_rgbd_frames(
            config.path_dataset)

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])