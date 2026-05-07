"""RoboBrain2.5 model wrapper for BaGLM.

RoboBrain2.5 checkpoints use ``model_type: qwen3_vl`` (Qwen3-VL backbone). The
inference path (processor + ``forward_multi_choice`` for VSG / progress) matches
``Qwen2VLModel`` in this repo, so we subclass it and only override ``load_model``.

Requires **transformers >= 4.57** (``Qwen3VLForConditionalGeneration``). The base
``requirements.txt`` pins ``transformers==4.49.0``; for RoboBrain install e.g.::

    pip install -r requirements_robobrain.txt
"""

import torch

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RoboBrain2.5 needs transformers>=4.57 with Qwen3VLForConditionalGeneration "
        "(checkpoint model_type is `qwen3_vl`). Install e.g.: "
        "pip install 'transformers>=4.57.0,<5.0'   or: pip install -r requirements_robobrain.txt"
    ) from e

from .qwen2vl_model import ModelPreprocessor, Qwen2VLModel

ROBOBRAIN_MODELS = {
    "robobrain2.5-4b": {
        "tokenizer": {
            "path": "/mnt/public1/dais/.cache/modelscope/hub/models/BAAI/RoboBrain2___5-4B",
        },
        "model": {
            "path": "/mnt/public1/dais/.cache/modelscope/hub/models/BAAI/RoboBrain2___5-4B",
            "torch_dtype": torch.bfloat16,
        },
    },
}


class RoboBrainModel(Qwen2VLModel):
    """RoboBrain2.5: same I/O as Qwen2.5-VL, loaded as Qwen3-VL."""

    video_mode = "direct"
    allows_image = True

    def __init__(self, model_name="robobrain2.5-4b", device="cuda", cache_dir=None):
        assert model_name in ROBOBRAIN_MODELS, (
            f"Model {model_name} not found in ROBOBRAIN_MODELS"
        )
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = ROBOBRAIN_MODELS[model_name]
        self.load_model()

    def get_preprocessor(self):
        return ModelPreprocessor(self)

    def load_model(self):
        model_cfg = self.model_info["model"]
        tokenizer_cfg = self.model_info["tokenizer"]
        dtype = model_cfg.get("torch_dtype", torch.bfloat16)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_cfg["path"],
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(tokenizer_cfg["path"])
        self.model.eval()

        self.device = next(self.model.parameters()).device
