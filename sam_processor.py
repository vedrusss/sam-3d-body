__author__ = "A.Antonenko, vedrusss@gmail.com"

import argparse
from time import time

import numpy as np
import torch

import _init_paths

from human_detector.human_detector_vitdet import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator


class SAM3D_Processor:
    def __init__(self, detector_folder_path: str, 
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
        self.__human_detector = HumanDetector(model_dir=detector_folder_path,  # куда сложить веса
                                              device=self.__device,
                                              download_if_missing=True,
                                              score_thresh=0.25)
        print("Person Detector is ready")
        print("Loading FOV Estimator")
        self.__fov_estimator = FOVEstimator(name="moge2", 
                                            device=self.__device,
                                            path=fov_path)
        print("FOV Estimator is ready")
        self.__estimator = SAM3DBodyEstimator(sam_3d_body_model=self.__model, model_cfg=self.__model_cfg,
                                              human_detector=self.__human_detector,
                                              human_segmentor=None,
                                              fov_estimator=self.__fov_estimator,
        )
        self.__bbox_thresh = 0.8
        self.__use_mask = False

    def __call__(self, image_path: str):
        outputs = self.__estimator.process_one_image(image_path,
                                                     bbox_thr=self.__bbox_thresh,
                                                     use_mask=self.__use_mask)
        return numpy_to_native(outputs[0])


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
