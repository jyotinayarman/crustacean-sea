"""
Pinned HuggingFace model revisions for reproducibility.

All external HuggingFace models/repos are pinned to specific commit hashes
to ensure reproducible builds as required by competition rules.
"""

HF_MODEL_REVISIONS: dict[str, str] = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    "Jyoti73ML/dinov3-lvd1689m": "fdad1b60473f919f1d238cc7ba2d0f042cb3efae",
    "ZhengPeng7/BiRefNet": "26e7919b869d089e5096e898c0492898f935604c",
    "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
    "lightx2v/Qwen-Image-Edit-2511-Lightning": "d74eba145674fd7e31b949324e148e21e7118abd",
}


def get_revision(repo_id: str) -> str | None:
    """Look up the pinned revision for a HuggingFace repo."""
    return HF_MODEL_REVISIONS.get(repo_id)
