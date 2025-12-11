# vitdet_cascade_h_inference.py
#
# Минимальный LazyConfig только для модели:
#   - base: common/models/mask_rcnn_vitdet.py
#   - cascade roi heads
#   - Huge ViT backbone
#
# Никаких train/dataloader/optimizer — только модель.

from functools import partial

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    CascadeROIHeads,
)

# 1. Базовый ViTDet Mask R-CNN конфиг (локальный файл из пакета detectron2)
_base = model_zoo.get_config("common/models/mask_rcnn_vitdet.py")

# 2. Берём только модель
model = _base.model

# (опционально) можно отключить сегментацию, если совсем не нужна:
# model.roi_heads.mask_on = False

# 3. Переводим ROIHeads в CascadeROIHeads

# удаляем аргументы, которые не совместимы с CascadeROIHeads
for k in ["box_head", "box_predictor", "proposal_matcher"]:
    if hasattr(model.roi_heads, k):
        model.roi_heads.pop(k)

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            # threshold переопределяем уже в коде загрузки
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# 4. Настройки Huge ViT backbone (ViTDet-H)

model.backbone.net.embed_dim = 1280
model.backbone.net.depth = 32
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.5

# глобальное внимание в нужных блоках
model.backbone.net.window_block_indexes = (
    list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
)

