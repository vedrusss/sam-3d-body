# human_detector_vitdet.py
#
# Простая обёртка над ViTDet Cascade Mask R-CNN (Huge) из Detectron2.
# - Один конфиг (локальный .py)
# - Один чекпоинт (model_final_f05665.pkl) в указанной директории
# - Явное кэширование весов без повторных загрузок из интернета

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import detectron2.data.transforms as T


_DEFAULT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/detectron2/"
    "ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(url: str, dst: Path) -> None:
    """
    Примитивный загрузчик файла с дискомфортно-простым логированием.
    Один раз скачали — дальше берём только локальный файл.
    """
    import requests

    dst_tmp = dst.with_suffix(dst.suffix + ".tmp")

    print(f"[HumanDetector] Downloading weights from {url} to {dst} ...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst_tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    dst_tmp.replace(dst)
    print("[HumanDetector] Download complete.")


class HumanDetector:
    """
    Обёртка для детектора людей на ViTDet Cascade Mask R-CNN (Huge).

    Основные идеи:
    - Все веса лежат в model_dir/model_final_f05665.pkl
    - Если файла нет, опционально скачиваем с официальный URL
    - Конфиг для модели — локальный python-файл vitdet_cascade_h_inference.py
    """

    def __init__(
        self,
        model_dir: str = "~/.cache/human_detector_vitdet",
        device: str = "cuda",
        download_if_missing: bool = True,
        score_thresh: float = 0.25,
    ):
        """
        :param model_dir: директория, где лежат/будут лежать веса.
        :param device: 'cuda' или 'cpu'.
        :param download_if_missing: если True, при отсутствии весов скачает их.
        :param score_thresh: порог score для боксов (будет применён к предикторам).
        """
        self.device = device
        self.model_dir = _ensure_dir(Path(model_dir).expanduser())
        self.checkpoint_path = self.model_dir / "model_final_f05665.pkl"

        # 1. Убедимся, что чекпоинт есть локально
        if not self.checkpoint_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.checkpoint_path}\n"
                    f"Скачай его вручную с:\n  {_DEFAULT_MODEL_URL}\n"
                    f"и положи по этому пути."
                )
            _download_file(_DEFAULT_MODEL_URL, self.checkpoint_path)

        # 2. Загрузим модель из локального lazy-конфига
        self.detector = self._load_detector(score_thresh=score_thresh)
        self.detector.to(self.device)
        self.detector.eval()

    def _load_detector(self, score_thresh: float):
        """
        Загрузка модели по локальному конфигу + локальным весам.
        Никаких detectron2:// и скрытых загрузок.
        """
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate, LazyConfig

        # Конфиг лежит рядом с этим файлом: vitdet_cascade_h_inference.py
        cfg_path = Path(__file__).parent / "vitdet_cascade_h_inference.py"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {cfg_path}\n"
                f"Убедись, что vitdet_cascade_h_inference.py лежит рядом с {__file__}."
            )

        cfg = LazyConfig.load(str(cfg_path))

        # Укажем путь к локальному чекпоинту (для загрузки весов)
        # Здесь train.* нам не нужен, но иногда удобно сохранить ссылку:
        if "train" in cfg and hasattr(cfg.train, "init_checkpoint"):
            cfg.train.init_checkpoint = str(self.checkpoint_path)

        # Настроим порог score для всех каскадных предикторов
        # (в ViTDet Cascade их обычно 3)
        if hasattr(cfg, "model") and hasattr(cfg.model, "roi_heads"):
            if hasattr(cfg.model.roi_heads, "box_predictors"):
                for p in cfg.model.roi_heads.box_predictors:
                    # test_score_thresh будет использоваться при inference
                    p.test_score_thresh = score_thresh

        # Инстанциируем модель
        model = instantiate(cfg.model)

        # Загрузим веса
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(str(self.checkpoint_path))

        model.eval()
        return model

    def run_human_detection(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = True,
    ) -> np.ndarray:
        """
        :param img: H x W x 3, np.uint8 или float32, BGR или RGB (в ViTDet RGB).
        :param det_cat_id: ID класса "person" в COCO (по умолчанию 0).
        :param bbox_thr: порог по score для фильтрации боксов.
        :param nms_thr: порог NMS (пока не используется, NMS внутри модели).
        :param default_to_full_image: если людей не нашли — вернуть [0,0,W,H].
        :return: np.ndarray [N, 4] (x1, y1, x2, y2)
        """
        return self.__run_detectron2_vitdet(
            img,
            det_cat_id=det_cat_id,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            default_to_full_image=default_to_full_image,
        )

    def __run_detectron2_vitdet(
        self,
        img: np.ndarray,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = True,
    ) -> np.ndarray:
        """
        Минимальный препроцесс + постпроцесс.
        Возвращаем только боксы (без масок).
        """
        height, width = img.shape[:2]

        IMAGE_SIZE = 1024
        transforms = T.ResizeShortestEdge(
            short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE
        )
        aug_input = T.AugInput(img)
        transforms(aug_input)
        img_transformed = aug_input.image  # уже трансформированное

        img_tensor = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        )

        inputs = {"image": img_tensor, "height": height, "width": width}

        with torch.no_grad():
            outputs = self.detector([inputs])

        instances = outputs[0]["instances"]
        # Фильтрация по классу и score
        valid_idx = (instances.pred_classes == det_cat_id) & (
            instances.scores > bbox_thr
        )

        if valid_idx.sum().item() == 0 and default_to_full_image:
            boxes = np.array([[0, 0, width, height]], dtype=np.float32)
        else:
            boxes = instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Сортируем боксы для детерминированного порядка
        if len(boxes) > 0:
            sorted_indices = np.lexsort(
                (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
            )
            boxes = boxes[sorted_indices]

        return boxes
