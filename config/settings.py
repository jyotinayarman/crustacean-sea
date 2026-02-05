from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent


class Settings(BaseSettings):
    api_title: str = "404 Base Miner Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    qwen_gpu: int = Field(default=0, env="QWEN_GPU")
    hunyuan_gpu: int = Field(default=0, env="HUNYUAN_GPU")
    dtype: str = Field(default="bf16", env="QWEN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Hunyuan3D settings (replaces Trellis for shape + texture)
    hunyuan_root: str = Field(default="", env="HUNYUAN_ROOT")
    hunyuan_model_path: str = Field(default="tencent/Hunyuan3D-2.1", env="HUNYUAN_MODEL_PATH")
    hunyuan_subfolder: str = Field(default="hunyuan3d-dit-v2-1", env="HUNYUAN_SUBFOLDER")
    hunyuan_save_dir: str = Field(default="", env="HUNYUAN_SAVE_DIR")
    hunyuan_mc_algo: str = Field(default="mc", env="HUNYUAN_MC_ALGO")
    hunyuan_enable_flashvdm: bool = Field(default=False, env="HUNYUAN_ENABLE_FLASHVDM")
    hunyuan_paint_bake_exp: int = Field(default=4, env="HUNYUAN_PAINT_BAKE_EXP")
    hunyuan_paint_reference_bg: str = Field(default="255,255,255", env="HUNYUAN_PAINT_REFERENCE_BG")

    # Qwen Edit settings
    qwen_edit_base_model_path: str = Field(
        default="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        env="QWEN_EDIT_BASE_MODEL_PATH"
    )
    qwen_edit_model_path: str = Field(
        default="Qwen/Qwen-Image-Edit-2511",
        env="QWEN_EDIT_MODEL_PATH"
    )
    qwen_edit_lora_repo: str = Field(
        default="lightx2v/Qwen-Image-Edit-2511-Lightning",
        env="QWEN_EDIT_LORA_REPO"
    )
    qwen_edit_height: int = Field(default=1024, env="QWEN_EDIT_HEIGHT")
    qwen_edit_width: int = Field(default=1024, env="QWEN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=4, env="NUM_INFERENCE_STEPS")
    true_cfg_scale: float = Field(default=1.0, env="TRUE_CFG_SCALE")
    qwen_edit_prompt_path: Path = Field(
        default=config_dir.joinpath("qwen_edit_prompt.json"),
        env="QWEN_EDIT_PROMPT_PATH"
    )

    # Multi-view generation prompts
    left_view_prompt: str = Field(
        default="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="LEFT_VIEW_PROMPT"
    )
    right_view_prompt: str = Field(
        default="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="RIGHT_VIEW_PROMPT"
    )
    back_view_prompt: str = Field(
        default="Show this object from the back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="BACK_VIEW_PROMPT"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]
