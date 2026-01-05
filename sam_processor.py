__author__ = "A.Antonenko, vedrusss@gmail.com"

import argparse
from time import time

import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union

from human_detector.human_detector_vitdet import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.utils import parse_pose_metainfo


class SAM3D_Processor:
    def __init__(self, detector_path: str, 
                       checkpoint_path: str, 
                       mhr_path: str, 
                       fov_path: str):
        # Initialize sam-3d-body model and other optional modules
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        print("Loading SAM-3D model")
        self.__model, self.__model_cfg = load_sam_3d_body(checkpoint_path, 
                                                          device=self.__device,
                                                          mhr_path=mhr_path)
        print("SAM 3D model is ready")
        print("Loading Person Detector")
        self.__human_detector = HumanDetector(model_path=detector_path,
                                              device=self.__device,
                                              download_if_missing=False,
                                              score_thresh=0.25)
        print("Person Detector is ready")
        print("Loading FOV Estimator")
        self.__fov_estimator = None # FOVEstimator(name="moge2", device=self.__device, path=fov_path)
        print("FOV Estimator is ready")
        self.__estimator = SAM3DBodyEstimator(sam_3d_body_model=self.__model, model_cfg=self.__model_cfg,
                                              human_detector=self.__human_detector,
                                              human_segmentor=None,
                                              fov_estimator=self.__fov_estimator,
        )
        self.__bbox_thresh = 0.8
        self.__use_mask = False

    def __call__(self, image: Union[str, np.ndarray], 
                 kp_scale: Optional[float]=None, box_scale: Optional[float]=None):
        #  image is either local path to an image or cv2 image (BGR format)
        outputs = self.__estimator.process_one_image(image,
                                                     bbox_thr=self.__bbox_thresh,
                                                     use_mask=self.__use_mask,
                                                     inference_type="body")
        return outputs, parse_sam_outputs_for_annotation_instances(outputs, kp_scale, box_scale)
        #return numpy_to_native(outputs[0]) if convert_from_numpy else outputs[0]


def numpy_to_native(obj):
    """
    Makes recursive convertion of all numpy-objects into JSON-serializable python types:
    - np.ndarray -> list
    - np.floating / np.integer / np.bool_ -> float/int/bool
    - dict, list, tuple, set -> checks entities
    The rest is returned as is
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()

    if isinstance(obj, dict):
        return {k: numpy_to_native(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [numpy_to_native(v) for v in obj]

    if isinstance(obj, set):
        return [numpy_to_native(v) for v in obj]

    # None, str, int, float, bool and rest -> as is
    return obj

def parse_sam_outputs_for_annotation_instances(
    outputs: Any,
    kp_scale: Optional[float] = None,
    box_scale: Optional[float] = None,
    tracking_policy: str = "zeros",  # "zeros" or "range"
):
    """
    SAM-3D-Body outputs (your real log format):
      outputs: list[dict], each dict has keys like:
        - "bbox": (4,) xyxy float32
        - "pred_keypoints_2d": (K,2) float32
        - "pred_keypoints_3d": (K,3) float32 (optional)
        - "mask": optional
        - etc.

    Returns: AnnotationFrame
    """

    def _as_np(a: Any, dtype=np.float32) -> Optional[np.ndarray]:
        if a is None:
            return None
        if isinstance(a, np.ndarray):
            return a.astype(dtype, copy=False)
        try:
            return np.asarray(a, dtype=dtype)
        except Exception:
            return None

    if outputs is None:
        return None

    def sam_kpts_to_movenet17(
        sam_kpts: np.ndarray,
        sam_to_movenet_idx: List[Optional[int]],
    ) -> np.ndarray:
        """
        Convert SAM keypoints to MoveNet-17 format by index mapping.

        Args:
            sam_kpts: np.ndarray of shape (K, D),
                      where D = 2 (2D) or D = 3 (3D)
            sam_to_movenet_idx: list of length 17,
                      each element is an index into sam_kpts or None

        Returns:
            np.ndarray of shape (17, D), dtype float32
        """
        if sam_kpts.ndim != 2:
            raise ValueError(f"sam_kpts must be 2D array, got shape {sam_kpts.shape}")

        K, D = sam_kpts.shape
        if D not in (2, 3):
            raise ValueError(f"Expected 2D or 3D keypoints, got D={D}")

        if len(sam_to_movenet_idx) != 17:
            raise ValueError("sam_to_movenet_idx must have length 17")

        out = np.zeros((17, D), dtype=np.float32)

        for mv_idx, sam_idx in enumerate(sam_to_movenet_idx):
            if sam_idx is None:
                continue
            if 0 <= sam_idx < K:
                out[mv_idx] = sam_kpts[sam_idx]

        return out

    # Normalize to list[dict]
    if isinstance(outputs, dict):
        det_list = [outputs]
    elif isinstance(outputs, (list, tuple)):
        det_list = list(outputs)
    else:
        # unknown container
        return None

    if len(det_list) == 0:
        return None

    if tracking_policy == "range":
        track_ids = list(range(len(det_list)))
    else:
        track_ids = [0] * len(det_list)

    sam_to_movenet_idx = build_sam_to_movenet_idx(parse_pose_metainfo(mhr70_pose_info))

    instances = []

    for i, det in enumerate(det_list):
        if not isinstance(det, dict):
            continue

        bbox = _as_np(det.get("bbox"), dtype=np.float32)  # expected (4,)
        kpts2d = _as_np(det.get("pred_keypoints_2d"), dtype=np.float32)  # expected (K,2)

        if bbox is None or kpts2d is None:
            continue

        bbox = bbox.squeeze()
        if bbox.shape != (4,):
            # if something odd (e.g. (1,4)), try reshape
            try:
                bbox = bbox.reshape(-1)
            except Exception:
                continue
            if bbox.shape[0] != 4:
                continue
            bbox = bbox[:4]

        # Ensure (K,2)
        if kpts2d.ndim == 3 and kpts2d.shape[0] == 1:
            kpts2d = kpts2d[0]
        if kpts2d.ndim != 2 or kpts2d.shape[1] != 2:
            continue

        # Reindex to MoveNet17
        mv_kpts2d = sam_kpts_to_movenet17(sam_kpts2d, sam_to_movenet_idx)  # (17,2)
        mv_scores = np.ones((17,), dtype=np.float32)

        # Apply your scaling conventions (as in YOLO converter)
        kp_xy_scaled = scale_obj(mv_kpts2d, kp_scale) if kp_scale is not None else mv_kpts2d
        bbox_scaled = scale_obj(bbox, box_scale) if box_scale is not None else bbox

        # Box confidence: в детекции нет явного score.
        # Минимально корректно — 1.0 (пока не определите метрику уверенности).
        box_conf = 1.0

        inst = dict(
            kp_xy_scaled=kp_xy_scaled,
            kp_scores=np.clip(mv_scores, 0, 1),
            bbox=bbox_scaled.squeeze(),
            box_confidence=float(box_conf),
            tracking_id=int(track_ids[i])
        )

        # Если хотите прикреплять 3D — адаптируйте под вашу модель данных:
        kpts3d = _as_np(det.get("pred_keypoints_3d"), dtype=np.float32)
        if kpts3d is not None:
            if kpts3d.ndim == 3 and kpts3d.shape[0] == 1:
                kpts3d = kpts3d[0]
            if kpts3d.ndim == 2 and kpts3d.shape[1] == 3:
                inst['kp_3d'] = sam_kpts_to_movenet17(kpts3d, sam_to_movenet_idx)

        instances.append(inst)

    return instances


MOVENET_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

def build_sam_to_movenet_idx(pose_meta_parsed: dict):
    """
    pose_meta_parsed = parse_pose_metainfo(...)
    returns: List[Optional[int]] length 17
    """
    name2id = pose_meta_parsed["keypoint_name2id"]

    def pick(*names):
        for n in names:
            if n in name2id:
                return name2id[n]
        return None

    return [
        pick("nose", "head", "neck"),                 # nose
        pick("left_eye", "left_ear", "head"),         # left_eye
        pick("right_eye", "right_ear", "head"),       # right_eye
        pick("left_ear", "head"),                     # left_ear
        pick("right_ear", "head"),                    # right_ear
        pick("left_shoulder"),
        pick("right_shoulder"),
        pick("left_elbow"),
        pick("right_elbow"),
        pick("left_wrist"),
        pick("right_wrist"),
        pick("left_hip"),
        pick("right_hip"),
        pick("left_knee"),
        pick("right_knee"),
        pick("left_ankle"),
        pick("right_ankle"),
    ]



def main(args):
    processor = SAM3D_Processor(args.detector, args.checkpoint_path, args.mhr_path)

    t0 = time()
    outputs = processor(args.image)
    print(outputs)
    print(f"Total time per image: {time() - t0} secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--detector", required=True, type=str, help="Path to folder with vetdet human detector model")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to SAM 3D Body model checkpoint")
    parser.add_argument("--mhr_path", default="", type=str, help="Path to MoHR/assets folder (or set SAM3D_mhr_path)")
    main(parser.parse_args())
