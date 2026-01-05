__author__ = "A.Antonenko, vedrusss@gmail.com"

from __future__ import annotations
import argparse
from time import time

import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union

from human_detector.human_detector_vitdet import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


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
    outputs: dict,
    kp_scale: Optional[float] = None,
    box_scale: Optional[float] = None,
    tracking_policy: str = "zeros",  # "zeros" or "range"
) -> List[dict]:
    """
    Convert SAM-3D-body `outputs` dict to 'anno' like instances.

    Expected keys (from your log):
      - bbox
      - pred_keypoints_2d
      - pred_keypoints_3d (optional)

    Notes:
      - Keypoint confidences are not provided in keys, so we set all kp scores to 1.0.
      - 3D keypoints are parsed but not stored unless your AnnotationInstance supports it.
    """

    def _as_np(a: Any, dtype=np.float32) -> np.ndarray:
        if isinstance(a, np.ndarray):
            return a.astype(dtype, copy=False)
        return np.asarray(a, dtype=dtype)

    def _ensure_N_first(x: np.ndarray, last_dim: int) -> np.ndarray:
        """
        Normalize either:
        - (last_dim,) -> (1,last_dim)
        - (N,last_dim) -> ok
        """
        x = np.asarray(x)
        if x.ndim == 1 and x.shape[0] == last_dim:
            return x[None, :]
        return x

    def _ensure_NK(x: np.ndarray, last_dim: int) -> np.ndarray:
        """
        Normalize either:
        - (K,last_dim) -> (1,K,last_dim)
        - (N,K,last_dim) -> ok
        """
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == last_dim:
            return x[None, :, :]
        return x

    if outputs is None or not isinstance(outputs, dict):
        # empty frame
        return None

    # --- BBoxes ---
    if "bbox" not in outputs or outputs["bbox"] is None:
        return None

    bboxes = _as_np(outputs["bbox"], dtype=np.float32)
    bboxes = _ensure_N_first(bboxes, last_dim=4)  # (N,4)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        # unexpected
        return None

    # --- 2D Keypoints ---
    if "pred_keypoints_2d" not in outputs or outputs["pred_keypoints_2d"] is None:
        return None

    kpts2d = _as_np(outputs["pred_keypoints_2d"], dtype=np.float32)
    kpts2d = _ensure_NK(kpts2d, last_dim=2)  # (N,K,2)
    if kpts2d.ndim != 3 or kpts2d.shape[2] != 2:
        return None

    # Optional: 3D Keypoints
    kpts3d = None
    if "pred_keypoints_3d" in outputs and outputs["pred_keypoints_3d"] is not None:
        kpts3d = _as_np(outputs["pred_keypoints_3d"], dtype=np.float32)
        kpts3d = _ensure_NK(kpts3d, last_dim=3)  # (N,K,3)
        if not (kpts3d.ndim == 3 and kpts3d.shape[2] == 3):
            kpts3d = None  # ignore if malformed

    # Align N across tensors
    N = min(bboxes.shape[0], kpts2d.shape[0])
    bboxes = bboxes[:N]
    kpts2d = kpts2d[:N]
    if kpts3d is not None:
        kpts3d = kpts3d[:N]

    # tracking ids
    if tracking_policy == "range":
        track_ids = list(range(N))
    else:
        track_ids = [0] * N

    instances = []
    for i in range(N):
        kp_xy = kpts2d[i]  # (K,2)
        kp_scores = np.ones((kp_xy.shape[0],), dtype=np.float32)  # no scores in outputs keys

        # scale like your YOLO converter
        kp_xy_scaled = scale_obj(kp_xy, kp_scale) if kp_scale is not None else kp_xy
        bbox_scaled = scale_obj(bboxes[i], box_scale) if box_scale is not None else bboxes[i]

        inst = dict(kp_xy_scaled=kp_xy_scaled, kp_scores=np.clip(kp_scores, 0, 1),
                    bbox=bbox_scaled.squeeze(), box_confidence=1.0,
                    tracking_id=int(track_ids[i]),
                    kp_3d = kpts3d[i])
        """
        inst = AnnotationInstance(
            kp_xy_scaled,
            np.clip(kp_scores, 0, 1),
            bbox=bbox_scaled.squeeze(),
            box_confidence=1.0,  # if you have a better confidence source, plug it here
            tracking_id=int(track_ids[i]),
            annotation_format=annotation_format,
        )
        """
        # If you want to attach 3D, adapt to your class fields:
        # if kpts3d is not None:
        #     inst.kp_3d = kpts3d[i]  # (K,3)

        instances.append(inst)

    return instances



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
