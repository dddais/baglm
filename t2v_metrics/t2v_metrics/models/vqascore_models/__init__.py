from typing import FrozenSet

from ...constants import HF_CACHE_DIR
from .internvl_model import INTERNVL2_MODELS, InternVL2Model
from .qwen2vl_model import QWEN2_VL_MODELS, Qwen2VLModel
try:
    from .robobrain_model import ROBOBRAIN_MODELS, RoboBrainModel
except ImportError:
    ROBOBRAIN_MODELS = {}
    RoboBrainModel = None

# Names only — do NOT import llavaov_model at package import time (it pulls in `llava`,
# which breaks on newer transformers, e.g. after upgrading for RoboBrain / Qwen3-VL).
_LLAVA_OV_MODEL_NAMES: FrozenSet[str] = frozenset(
    {
        "llava-onevision-qwen2-7b-si",
        "llava-onevision-qwen2-7b-ov",
    }
)


def list_all_vqascore_models():
    # Always advertise LLaVA names; actual import happens only in get_vqascore_model.
    # Probing import here used to run `import llava` and spam stderr on broken stacks.
    return (
        list(_LLAVA_OV_MODEL_NAMES)
        + list(INTERNVL2_MODELS)
        + list(QWEN2_VL_MODELS)
        + list(ROBOBRAIN_MODELS)
    )


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models(), f"Unknown VQA model: {model_name}"

    if model_name in _LLAVA_OV_MODEL_NAMES:
        try:
            from .llavaov_model import LLaVAOneVisionModel
        except ImportError as e:
            raise ImportError(
                "LLaVA-OneVision is not available with your current `transformers` + `llava` "
                "combination (often `apply_chunking_to_forward` removed after transformers>=4.50). "
                "Use `internvl*` / `qwen2.5-vl*` / `robobrain*` models, or install a compatible pair "
                "in a separate environment."
            ) from e
        return LLaVAOneVisionModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    if model_name in INTERNVL2_MODELS:
        return InternVL2Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    if model_name in QWEN2_VL_MODELS:
        return Qwen2VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    if model_name in ROBOBRAIN_MODELS:
        return RoboBrainModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError()
