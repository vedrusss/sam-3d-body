# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict

def safe_load_state_dict(model: torch.nn.Module, state: dict, *, verbose: bool = True):
    """
    Аккуратная загрузка весов:
    - strict=False, но
    - мы явно контролируем, какие missing/unexpected ключи допустимы
    """
    incompatible = model.load_state_dict(state, strict=False)

    # PyTorch возвращает IncompatibleKeys(missing_keys=[...], unexpected_keys=[...])
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    # Разрешённые префиксы для "инфраструктурных" ключей,
    # которые по дизайну могут не лежать в чекпоинте
    allowed_missing_prefixes = (
        "backbone.encoder.mask_token",
        "head_pose.",
        "head_pose_hand.",
        "head_pose.mhr.",
        "head_pose_hand.mhr.",
        "head_pose.mhr.character_torch.",
        "head_pose_hand.mhr.character_torch.",
    )

    def _filter_allowed(keys, prefixes):
        return [
            k
            for k in keys
            if all(not k.startswith(p) for p in prefixes)
        ]

    real_missing = _filter_allowed(missing, allowed_missing_prefixes)

    if verbose:
        if missing or unexpected:
            print("[safe_load_state_dict] Incompatible keys:")
            if missing:
                print("  missing:", missing)
            if unexpected:
                print("  unexpected:", unexpected)

    # Если есть "настоящие" критичные missing-keys — падаем.
    if real_missing:
        raise RuntimeError(
            "Критичные missing_keys при загрузке state_dict:\n"
            + "\n".join(f"  - {k}" for k in real_missing)
        )

    return incompatible


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):  
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    safe_load_state_dict(model, state_dict, verbose=False)
    #load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
