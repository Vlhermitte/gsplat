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
    ):
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")
        self.colmap_model = Reconstruction(colmap_dir + "/sparse/0")
        self.image_dir = image_dir
        self.split = split
        self.patch_size = patch_size
        self.factor = factor
        self.test_every = test_every
        self.normalize = normalize
        self.image_names = []
        self.image_paths = sorted(_get_rel_paths(image_dir))
        self.camtoworlds = []

        for image in self.colmap_model.images.values():
            if image.registered:
                self.image_names.append(image.name)
                w2c = image.cam_from_world.matrix()
                w2c = np.vstack((w2c, np.array([[0, 0, 0, 1]])))
                self.camtoworlds.append(w2c)
        self.image_names = sorted(self.image_names)
        self.camtoworlds = np.linalg.inv(np.array(self.camtoworlds)) # We apply the inverse here

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
            self._fill_in_missing_poses()
            indices = np.arange(len(self.colmap_model.images))
            self.indices = indices[indices % test_every == 0]

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
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
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
                    print(f"Image {image_name} not registered. Using {closest_image} instead.")
                    borrowed_pycolmap_image = copy.deepcopy(self.colmap_model.find_image_with_name(closest_image))
                    borrowed_pycolmap_image.name = image_name
                    borrowed_pycolmap_image.image_id = max(self.colmap_model.images.keys()) + 1
                    borrowed_pycolmap_image.registered = False
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
            return np.array([camera.params[4], 0.0, 0.0, 0.0], dtype=np.float32)
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