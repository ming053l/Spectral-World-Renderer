#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Camera, image and HSI loading of a scene for stage `pack`, from the 3D Gaussian Splatting scene loader (Inria).

COLMAP and Blender camera readers (scene/colmap_loader.py, scene/dataset_readers.py), graphics helpers and
`load_camera` (resolution, RGB, alpha, HSI and the world-to-view matrix, as utils/camera_utils.py and
scene/cameras.py). The 3DGS model, depth maps and point clouds are not needed for a pack and are left out;
`Camera` of colmap_loader is renamed `ColmapCamera`.
"""
import collections
import json
import math
import os
import struct
import sys
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image as PILImage

from .data import load_hsi_numpy, resize_hsi_tensor, resolve_hsi_path


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])


ColmapCamera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])


BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}


CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = ColmapCamera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = ColmapCamera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def _subsample_train_cameras(train_cam_infos, train_view_fraction):
    if train_view_fraction is None or train_view_fraction >= 0.999:
        return train_cam_infos
    if train_view_fraction <= 0:
        raise ValueError(f"train_view_fraction must be positive, got {train_view_fraction}")
    n_cameras = len(train_cam_infos)
    if n_cameras <= 1:
        return train_cam_infos
    n_keep = max(1, int(round(n_cameras * train_view_fraction)))
    n_keep = min(n_keep, n_cameras)
    indices = np.linspace(0, n_cameras - 1, n_keep)
    keep = sorted(set(int(round(idx)) for idx in indices))
    selected = [train_cam_infos[idx] for idx in keep]
    print(f"Using train_view_fraction={train_view_fraction:.3f}: {len(selected)}/{n_cameras} training cameras")
    return selected


def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = PILImage.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = PILImage.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))

    return cam_infos


def read_cameras(source_path, images, white_background, eval, train_test_exp=False, train_view_fraction=1.0, llffhold=8):
    """(TRAIN, TEST) CameraInfo lists as Scene.__init__ builds them (no depth maps, no point cloud)."""
    if os.path.exists(os.path.join(source_path, "sparse")):
        return _read_colmap(source_path, images, eval, train_test_exp, train_view_fraction, llffhold)
    if os.path.exists(os.path.join(source_path, "transforms_train.json")):
        train = readCamerasFromTransforms(source_path, "transforms_train.json", "", white_background, False, ".png")
        test = readCamerasFromTransforms(source_path, "transforms_test.json", "", white_background, True, ".png")
        if not eval:
            train.extend(test)
            test = []
        return train, test
    raise ValueError("Could not recognize scene type: " + str(source_path))


def _read_colmap(path, images, eval, train_test_exp, train_view_fraction, llffhold):
    """readColmapSceneInfo without depth parameters and point cloud."""
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))
    if eval:
        if "360" in path:
            llffhold = 8
        fixed_manifest = os.path.join(path, "fixed_test_views.json")
        if os.path.isfile(fixed_manifest):
            with open(fixed_manifest) as f:
                test_cam_names_list = json.load(f)
            all_names = {c.name for c in cam_extrinsics.values()}
            chosen = set(test_cam_names_list)
            if not chosen or len(chosen) != len(test_cam_names_list) or not chosen < all_names:
                raise ValueError("Invalid explicit held-out camera manifest")
        elif llffhold:
            cam_names = sorted(cam_extrinsics[cam_id].name for cam_id in cam_extrinsics)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=None,
                                           images_folder=os.path.join(path, reading_dir), depths_folder="",
                                           test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    return _subsample_train_cameras(train_cam_infos, float(train_view_fraction)), test_cam_infos


def load_camera(args, cam_info, is_test_dataset):
    """camera_utils.loadCam + cameras.Camera (image, alpha mask, HSI, FoV, world-to-view) for resolution scale 1."""
    image = PILImage.open(cam_info.image_path)
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / args.resolution), round(orig_h / args.resolution)
    else:
        if args.resolution == -1:
            global_down = orig_w / 1600 if orig_w > 1600 else 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down) * float(1.0)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    hsi_image = None
    hsi_path = resolve_hsi_path(cam_info.image_path, source_path=args.source_path, hsi_dir=args.hsi_dir)
    if hsi_path is not None:
        hsi_image = resize_hsi_tensor(load_hsi_numpy(hsi_path), resolution)
    resized = PILtoTorch(image, resolution)
    alpha_mask = resized[3:4, ...] if resized.shape[0] == 4 else torch.ones_like(resized[0:1, ...])
    if args.train_test_exp and cam_info.is_test:
        if is_test_dataset:
            alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
        else:
            alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0
    original_image = resized[:3, ...].clamp(0.0, 1.0)
    return Namespace(image_name=cam_info.image_name, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                     original_image=original_image, alpha_mask=alpha_mask,
                     original_hsi=hsi_image.clamp(0.0, 1.0) if hsi_image is not None else None,
                     image_width=original_image.shape[2], image_height=original_image.shape[1],
                     world_view_transform=torch.tensor(getWorld2View2(cam_info.R, cam_info.T)).transpose(0, 1))


def load_scene(saved, anchor):
    """(args, TRAIN cameras, TEST cameras) of the scene described by the 'args' dict of a reference state file."""
    args = Namespace(**saved)
    args.model_path = str(anchor)
    args.source_path = os.path.abspath(args.source_path)
    args.eval = True
    train, test = read_cameras(args.source_path, getattr(args, 'images', 'images'), getattr(args, 'white_background', False),
                               True, getattr(args, 'train_test_exp', False), getattr(args, 'train_view_fraction', 1.0))
    return (args, [load_camera(args, c, False) for c in train], [load_camera(args, c, True) for c in test])
