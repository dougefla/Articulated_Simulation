from math import cos, sin
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c

from s2u.utils.transform import Transform
import torch


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

# class TSDFVolume(object):
#     """Integration of multiple depth images using a TSDF."""

#     def __init__(self, size, resolution, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor):
#         self.size = size
#         self.resolution = resolution
#         self.voxel_size = self.size / self.resolution
#         self.sdf_trunc = 4 * self.voxel_size

#         self._volume = o3d.pipelines.integration.UniformTSDFVolume(
#             length=self.size,
#             resolution=self.resolution,
#             sdf_trunc=self.sdf_trunc,
#             color_type=color_type,
#         )

#     def integrate(self, depth_img, intrinsic, extrinsic):
#         """
#         Args:
#             depth_img: The depth image.
#             intrinsic: The intrinsic parameters of a pinhole camera model.
#             extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
#         """
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d.geometry.Image(np.empty_like(depth_img)),
#             o3d.geometry.Image(depth_img),
#             depth_scale=1.0,
#             depth_trunc=2.0,
#             convert_rgb_to_intensity=False,
#         )
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(
#             width=intrinsic.width,
#             height=intrinsic.height,
#             fx=intrinsic.fx,
#             fy=intrinsic.fy,
#             cx=intrinsic.cx,
#             cy=intrinsic.cy,
#         )
#         extrinsic = extrinsic.as_matrix()
#         self._volume.integrate(rgbd, intrinsic, extrinsic)

#     def integrate_rgb(self, depth_img, rgb, intrinsic, extrinsic):
#         """
#         Args:
#             depth_img: The depth image.
#             intrinsic: The intrinsic parameters of a pinhole camera model.
#             extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
#         """
#         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             o3d.geometry.Image(rgb),
#             o3d.geometry.Image(depth_img),
#             depth_scale=1.0,
#             depth_trunc=2.0,
#             convert_rgb_to_intensity=False,
#         )
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(
#             width=intrinsic.width,
#             height=intrinsic.height,
#             fx=intrinsic.fx,
#             fy=intrinsic.fy,
#             cx=intrinsic.cx,
#             cy=intrinsic.cy,
#         )
#         extrinsic = extrinsic.as_matrix()
#         self._volume.integrate(rgbd, intrinsic, extrinsic)

#     def get_grid(self):
#         # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
#         shape = (1, self.resolution, self.resolution, self.resolution)
#         tsdf_grid = np.zeros(shape, dtype=np.float32)
#         voxels = self._volume.extract_voxel_grid().get_voxels()
#         for voxel in voxels:
#             i, j, k = voxel.grid_index
#             tsdf_grid[0, i, j, k] = voxel.color[0]
#         return tsdf_grid

#     def get_cloud(self):
#         return self._volume.extract_point_cloud()

class TSDFVolume(object):
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution, label_channel = 4):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size
        self.depth_scale = 10
        self.depth_max = 3.0
        self.device = o3d.core.Device("CUDA:0")
        self.label_channel = label_channel

        self._volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names = ('tsdf', 'weight', 'label'),
            attr_dtypes = (o3c.float32, o3c.float32, o3c.float32),
            attr_channels = ((1), (1), (self.label_channel)), 
            voxel_size = self.size / self.resolution,
            block_resolution = 8,
            block_count = 10000,
            device = self.device
        )
    
    def integrate(self, depth_img, label_img, intrinsic, extrinsic):

        trunc = self.sdf_trunc

        depth = o3d.t.geometry.Image(depth_img).to(self.device)

        intrinsic = o3c.Tensor(o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        ).intrinsic_matrix)
        extrinsic = o3c.Tensor(extrinsic.as_matrix())
        # Get active frustum block coordinates from input
        frustum_block_coords = self._volume.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, self.depth_scale,
            self.depth_max)
        # Activate them in the underlying hash map (may have been inserted)
        self._volume.hashmap().activate(frustum_block_coords)

        # Find buf indices in the underlying engine
        buf_indices, masks = self._volume.hashmap().find(frustum_block_coords)
        o3d.core.cuda.synchronize()
        end = time.time()

        start = time.time()
        voxel_coords, voxel_indices = self._volume.voxel_coordinates_and_flattened_indices(
            buf_indices)
        o3d.core.cuda.synchronize()
        end = time.time()

        # Now project them to the depth and find association
        # (3, N) -> (2, N)
        start = time.time()
        extrinsic_dev = extrinsic.to(self.device, o3c.float32)
        xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                        3:]

        intrinsic_dev = intrinsic.to(self.device, o3c.float32)
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
            o3c.float32) / self.depth_scale
        sdf = depth_readings - d_proj

        mask_inlier = (depth_readings > 0) \
            & (depth_readings < self.depth_max) \
            & (sdf >= -trunc)

        sdf[sdf >= trunc] = trunc
        sdf = sdf / trunc
        o3d.core.cuda.synchronize()
        end = time.time()

        start = time.time()
        weight = self._volume.attribute('weight').reshape((-1, 1))
        tsdf = self._volume.attribute('tsdf').reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        w = weight[valid_voxel_indices]
        wp = w + 1

        tsdf[valid_voxel_indices] \
            = (tsdf[valid_voxel_indices] * w +
                sdf[mask_inlier].reshape(w.shape)) / (wp)
        
        # label_readings = o3c.Tensor(label_img).to(self.device,o3c.float32)
        # label_readings = label_readings[v_proj,u_proj]
        # label = self._volume.attribute('label').reshape((-1, self.label_channel))
        # label[valid_voxel_indices] \
        #     = (label[valid_voxel_indices] * w +
        #                 label_readings[mask_inlier]) / (wp) 

        weight[valid_voxel_indices] = wp
        o3d.core.cuda.synchronize()
        end = time.time()
    

    # def integrate(self, depth_img, intrinsic, extrinsic):
    #     """
    #     Args:
    #         depth_img: The depth image.
    #         intrinsic: The intrinsic parameters of a pinhole camera model.
    #         extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
    #     """
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         o3d.geometry.Image(np.empty_like(depth_img)),
    #         o3d.geometry.Image(depth_img),
    #         depth_scale=1.0,
    #         depth_trunc=2.0,
    #         convert_rgb_to_intensity=False,
    #     )
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         width=intrinsic.width,
    #         height=intrinsic.height,
    #         fx=intrinsic.fx,
    #         fy=intrinsic.fy,
    #         cx=intrinsic.cx,
    #         cy=intrinsic.cy,
    #     )
    #     extrinsic = extrinsic.as_matrix()
    #     self._volume.integrate(rgbd, intrinsic, extrinsic)

    # def integrate_rgb(self, depth_img, rgb, intrinsic, extrinsic):
    #     """
    #     Args:
    #         depth_img: The depth image.
    #         intrinsic: The intrinsic parameters of a pinhole camera model.
    #         extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
    #     """
    #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         o3d.geometry.Image(rgb),
    #         o3d.geometry.Image(depth_img),
    #         depth_scale=1.0,
    #         depth_trunc=2.0,
    #         convert_rgb_to_intensity=False,
    #     )
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         width=intrinsic.width,
    #         height=intrinsic.height,
    #         fx=intrinsic.fx,
    #         fy=intrinsic.fy,
    #         cx=intrinsic.cx,
    #         cy=intrinsic.cy,
    #     )
    #     extrinsic = extrinsic.as_matrix()
    #     self._volume.integrate(rgbd, intrinsic, extrinsic)

    # def get_grid(self):
    #     # TODO(mbreyer) very slow (~35 ms / 50 ms of the whole pipeline)
    #     shape = (1, self.resolution, self.resolution, self.resolution)
    #     tsdf_grid = np.zeros(shape, dtype=np.float32)
    #     voxels = self._volume.extract_voxel_grid().get_voxels()
    #     for voxel in voxels:
    #         i, j, k = voxel.grid_index
    #         tsdf_grid[0, i, j, k] = voxel.color[0]
    #     return tsdf_grid

    # def get_cloud(self):
    #     return self._volume.extract_point_cloud()

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