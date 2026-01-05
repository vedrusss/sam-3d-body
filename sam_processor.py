__author__ = "A.Antonenko, vedrusss@gmail.com"

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

        # Keypoint scores отсутствуют — ставим 1.0 (или можно вывести суррогат, см. ниже)
        kp_scores = np.ones((kpts2d.shape[0],), dtype=np.float32)

        # Apply your scaling conventions (as in YOLO converter)
        kp_xy_scaled = scale_obj(kpts2d, kp_scale) if kp_scale is not None else kpts2d
        bbox_scaled = scale_obj(bbox, box_scale) if box_scale is not None else bbox

        # Box confidence: в детекции нет явного score.
        # Минимально корректно — 1.0 (пока не определите метрику уверенности).
        box_conf = 1.0

        inst = dict(
            kp_xy_scaled=kp_xy_scaled,
            kp_scores=np.clip(kp_scores, 0, 1),
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
                inst['kp_3d'] = kpts3d

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
