import copy
import os
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never
from collections import defaultdict
import cv2
from PIL import Image
import imageio.v2 as imageio
import numpy as np
import torch
from difflib import SequenceMatcher
from pycolmap import CameraModelId, Reconstruction

if __name__ == "__main__":
    from normalize import (
        align_principal_axes,
        similarity_from_cameras,
        transform_cameras,
        transform_points,
    )
else:
    from .normalize import (
        align_principal_axes,
        similarity_from_cameras,
        transform_cameras,
        transform_points,
    )


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        # resized_path = os.path.join(
        #     resized_dir, os.path.splitext(image_file)[0] + ".png"
        # )
        resized_path = os.path.join(
            resized_dir, image_file
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _part_distance(a: str, b: str) -> float:
    """Compute a distance between two filename-parts."""
    try:
        # numeric distance
        return abs(int(a) - int(b))
    except ValueError:
        # non-numeric: use a simple 0/1 or a fuzzy ratio
        if a == b:
            return 0.0
        # or, for a smoother measure:
        return 1.0 - SequenceMatcher(None, a, b).ratio()


def _find_closest_image(src_image: str, image_list: List[str]) -> str:
    """Find the closest image in the list to the source image."""
    src_base = os.path.splitext(os.path.basename(src_image))[0]
    src_parts = src_base.split("_")

    best_match = None
    best_score = float("inf")

    for img in image_list:
        cand_base = os.path.splitext(os.path.basename(img))[0]
        cand_parts = cand_base.split("_")
        # zip will drop extra parts; if your names vary in part‐counts you may want to pad
        score = sum(
            _part_distance(s, c)
            for s, c in zip(src_parts, cand_parts)
        )
        if score < best_score:
            best_score = score
            best_match = img

    return best_match


class Parser:
    """COLMAP parser."""

    def __init__(
            self,
            data_dir: str,
            factor: int = 1,
            normalize: bool = False,
            test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        colmap_image_dir = os.path.join(data_dir, "images")
        self.image_list = sorted(os.listdir(colmap_image_dir))
        self.colmap_model = Reconstruction(colmap_dir)  # model from the colmap directory

        point3D_id_contiguous = dict()
        for i, point_id in enumerate(self.colmap_model.points3D.keys()):
            point3D_id_contiguous[point_id] = i

        # Extract extrinsic matrices in world-to-camera format.
        imdata = self.colmap_model.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        point_indices = defaultdict(list)
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            w2c = im.cam_from_world.matrix()
            w2c = np.concatenate([w2c, bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = self.colmap_model.cameras[camera_id]
            K = cam.calibration_matrix()
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.model
            if type_ == CameraModelId.SIMPLE_PINHOLE:
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == CameraModelId.PINHOLE:
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == CameraModelId.SIMPLE_RADIAL:
                params = np.array([cam.params[3], 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == CameraModelId.RADIAL:
                params = np.array([cam.params[3], cam.params[4], 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == CameraModelId.OPENCV:
                params = np.array([cam.params[3], cam.params[4], cam.params[5], cam.params[6]], dtype=np.float32)
                camtype = "perspective"
            elif type_ == CameraModelId.OPENCV_FISHEYE:
                params = np.array([cam.params[3], cam.params[4], cam.params[5], cam.params[6]], dtype=np.float32)
                camtype = "fisheye"
            assert (
                    camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None

            for obs_point2d in im.get_valid_points2D():
                point_indices[im.name].append(point3D_id_contiguous[obs_point2d.point3D_id])

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == CameraModelId.SIMPLE_PINHOLE or type_ == CameraModelId.PINHOLE):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points3D = self.colmap_model.points3D.values()
        points_err = np.array([p.error for p in points3D])
        points_rgb = np.array([p.color for p in points3D])
        points = np.array([p.xyz for p in points3D])

        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                    camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1 ** 2 + y1 ** 2)
                r = (
                        1.0
                        + params[0] * theta ** 2
                        + params[1] * theta ** 4
                        + params[2] * theta ** 6
                        + params[3] * theta ** 8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
            self,
            colmap_dir: str,
            image_dir: str,
            split: str = "train",
            patch_size: Optional[int] = None,
            test_every: int = 8,
            factor: int = 4,
            normalize: bool = True,
            load_depths: bool = False,
    ):
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")
        self.colmap_model = None
        self.colmap_dir = colmap_dir + "/sparse/0"
        self.image_dir = image_dir
        self.split = split
        self.patch_size = patch_size
        self.factor = factor
        self.test_every = test_every
        self.normalize = normalize
        self.image_names = []
        self.image_paths = sorted(_get_rel_paths(image_dir))
        self.camtoworlds = []

        self._build_reconstruction()

        for image in self.colmap_model.images.values():
            if image.registered:
                self.image_names.append(image.name)
                w2c = image.cam_from_world.matrix()
                w2c = np.vstack((w2c, np.array([[0, 0, 0, 1]])))
                self.camtoworlds.append(w2c)
        self.image_names = sorted(self.image_names)
        self.camtoworlds = np.linalg.inv(np.array(self.camtoworlds))  # We apply the inverse here

        # points3D
        points3D = self.colmap_model.points3D.values()
        if len(points3D) == 0:
            print("Warning: No points3D found in the COLMAP model.")
        self.points_rgb = np.array([p.color for p in points3D])
        self.points = np.array([p.xyz for p in points3D])

        # Normalize the world space.
        if self.normalize and len(self.points) > 0:
            T1 = similarity_from_cameras(self.camtoworlds)
            self.camtoworlds = transform_cameras(T1, self.camtoworlds)
            self.points = transform_points(T1, self.points)

            T2 = align_principal_axes(self.points)
            self.camtoworlds = transform_cameras(T2, self.camtoworlds)
            self.points = transform_points(T2, self.points)

            self.transform = T2 @ T1
        else:
            self.transform = np.eye(4)

        if self.factor > 1:
            self.image_dir = _resize_image_folder(
                image_dir=image_dir, resized_dir=self.image_dir + f"_{factor}", factor=factor
            )
            self.image_paths = sorted(_get_rel_paths(image_dir))

        if split == "train":
            indices = np.arange(len(self.colmap_model.images))
            self.indices = indices[indices % test_every != 0]
        else:
            indices = np.arange(len(self.colmap_model.images))
            self.indices = indices[indices % test_every == 0]

    def _build_reconstruction(self):
        self.colmap_model = Reconstruction(self.colmap_dir)
        if self.split != "train":
            self._fill_in_missing_poses()

    def __getstate__(self):
        st = self.__dict__.copy()
        # drop the heavy C++ object
        st.pop("colmap_model", None)
        return st

    def __setstate__(self, state):
        # restore everything else
        self.__dict__.update(state)
        # lazily rebuild the Reconstruction in each worker
        self._build_reconstruction()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_name = self.image_names[index]
        colmap_image = self.colmap_model.find_image_with_name(image_name)
        image = imageio.imread(os.path.join(self.image_dir, image_name))[..., :3]
        camera_id = colmap_image.camera_id
        camera = self.colmap_model.cameras[camera_id]
        K = camera.calibration_matrix()
        K[:2, :] /= self.factor
        params = self._get_camera_params(camera)
        w2c = colmap_image.cam_from_world.matrix()
        w2c = np.vstack((w2c, np.array([[0, 0, 0, 1]])))
        c2w = np.linalg.inv(w2c)

        if len(params) > 0:
            # Images are distorted. Undistort them. For now does not support Fisheye cameras
            width = int(camera.width // self.factor)
            height = int(camera.height // self.factor)
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )

            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi_undist
            image = image[y: y + h, x: x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y: y + self.patch_size, x: x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(c2w).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        return data

    def _fill_in_missing_poses(self):
        image_list = sorted(os.listdir(self.image_dir))
        registered_images_name = []
        for image in self.colmap_model.images.values():
            if image.registered:
                registered_images_name.append(image.name)

        for image_name in image_list:
            if image_name not in registered_images_name:
                # Find the closest image in the list to the source image.
                closest_image = _find_closest_image(
                    src_image=image_name, image_list=registered_images_name
                )
                if closest_image:
                    borrowed_pycolmap_image = copy.copy(self.colmap_model.find_image_with_name(closest_image))
                    borrowed_pycolmap_image.name = image_name
                    borrowed_pycolmap_image.image_id = max(self.colmap_model.images.keys()) + 1
                    borrowed_pycolmap_image.registered = True
                    if borrowed_pycolmap_image.image_id in self.colmap_model.images:
                        raise ValueError(f"Image {image_name} already exists in the model.")
                    self.colmap_model.add_image(borrowed_pycolmap_image)

    @staticmethod
    def _get_camera_params(camera):
        type = camera.model
        if type == CameraModelId.SIMPLE_PINHOLE or type == "SIMPLE_PINHOLE":
            return np.empty(0, dtype=np.float32)
        elif type == CameraModelId.PINHOLE or type == "PINHOLE":
            return np.empty(0, dtype=np.float32)
        if type == CameraModelId.SIMPLE_RADIAL or type == "SIMPLE_RADIAL":
            return np.array([camera.params[3], 0.0, 0.0, 0.0], dtype=np.float32)
        elif type == CameraModelId.RADIAL or type == "RADIAL":
            return np.array([camera.params[3], camera.params[4], 0.0, 0.0], dtype=np.float32)
        elif type == CameraModelId.OPENCV or type == "OPENCV":
            return np.array([camera.params[3], camera.params[4], camera.params[5], camera.params[6]], dtype=np.float32)
        else:
            raise ValueError(f"Camera model {type} not supported.")


if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../../data/results/glomap/ETH3D/courtyard/colmap")
    parser.add_argument("--images_dir", type=str, default="../../../data/datasets/ETH3D/courtyard/images")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    train_dataset = Dataset(args.data_dir, args.images_dir, split="train", factor=args.factor)
    eval_dataset = Dataset(args.data_dir, args.images_dir, split="eval", factor=args.factor)
    print(f"Train Dataset: {len(train_dataset)} images.")
    print(f"Eval Dataset: {len(eval_dataset)} images.")

    train_img = [img for img in train_dataset]
    eval_img = [img for img in eval_dataset]

    print("Done")