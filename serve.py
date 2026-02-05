import gc
import os
import sys
import argparse
import asyncio
from io import BytesIO
from time import time
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
import psutil
from loguru import logger
from fastapi import FastAPI, UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

from config import settings
from modules.image_edit import QwenEditModule
from modules.rembg_birefnet import BiRefNetPreprocess
from utils_glb import normalize_glb_to_unit_cube


def _hunyuan_paint_dir():
    return os.path.join(os.path.dirname(__file__), "modules", "hunyuan", "paint")


def _hunyuan_save_dir():
    if getattr(settings, "hunyuan_save_dir", "") and str(settings.hunyuan_save_dir).strip():
        return os.path.abspath(str(settings.hunyuan_save_dir))
    return os.path.join(os.path.dirname(__file__), "hunyuan_cache")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    mem = psutil.virtual_memory()
    logger.info(f"STARTUP: Memory available: {mem.available / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB")
    sys.stdout.flush()
    try:
        from modules.hunyuan.paint.utils.torchvision_fix import apply_fix
        apply_fix()
    except Exception:
        pass

    try:
        logger.info("Loading Hunyuan3D shape pipeline...")
        sys.stdout.flush()
        torch.cuda.empty_cache()
        gc.collect()
        mem = psutil.virtual_memory()
        logger.info(f"Before checkpoint load: {mem.available / 1024**3:.1f}GB free")
        sys.stdout.flush()
        import modules.hunyuan.shape as _hy3dshape
        sys.modules["hy3dshape"] = _hy3dshape
        from modules.hunyuan import Hunyuan3DDiTFlowMatchingPipeline
        
        # Check if safetensors exists, fallback to ckpt if not
        from modules.hunyuan.shape.utils import smart_load_model
        _, ckpt_path_safetensors = smart_load_model(
            settings.hunyuan_model_path,
            settings.hunyuan_subfolder,
            use_safetensors=True,
            variant="fp16"
        )
        use_safetensors = os.path.exists(ckpt_path_safetensors)
        
        if not use_safetensors:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mem = psutil.virtual_memory()
            logger.warning(
                f"Safetensors format not found. Using .ckpt format which loads ~7GB at once. "
                f"Current free RAM: {mem.available / 1024**3:.1f}GB. "
                f"OOM 'Killed' is common without enough RAM+swap. "
                f"Options: (1) Add 8GB+ swap; (2) Convert to safetensors on a 16GB+ machine"
            )
            sys.stdout.flush()
        logger.info(f"Loading checkpoint (safetensors={use_safetensors})...")
        sys.stdout.flush()
        app.state.hunyuan_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            settings.hunyuan_model_path,
            use_safetensors=use_safetensors
        )
        mem = psutil.virtual_memory()
        logger.info(f"After checkpoint load: {mem.available / 1024**3:.1f}GB free")
        sys.stdout.flush()
        if getattr(settings, "hunyuan_enable_flashvdm", False):
            mc = getattr(settings, "hunyuan_mc_algo", "mc")
            app.state.hunyuan_shape.enable_flashvdm(mc_algo=mc)
        logger.info("Hunyuan shape pipeline loaded")
    except Exception as e:
        logger.exception(f"Hunyuan shape loading: {e}")
        raise SystemExit("Hunyuan shape failed to load")

    try:
        logger.info("Loading BiRefNet (rembg, same as before)...")
        torch.cuda.empty_cache()
        gc.collect()
        device = f"cuda:{settings.hunyuan_gpu}" if torch.cuda.is_available() else "cpu"
        app.state.birefnet_preprocess = BiRefNetPreprocess(device=device)
        if torch.cuda.is_available():
            app.state.birefnet_preprocess.model
        logger.info("BiRefNet loaded")
    except Exception as e:
        logger.exception(f"BiRefNet loading: {e}")
        raise SystemExit("BiRefNet failed to load")

    try:
        logger.info("Loading Hunyuan3D paint pipeline...")
        torch.cuda.empty_cache()
        gc.collect()
        from modules.hunyuan.paint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        paint_dir = _hunyuan_paint_dir()
        max_num_view = 6
        resolution = 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        realesrgan_path = os.path.join(paint_dir, "ckpt", "RealESRGAN_x4plus.pth")
        if not os.path.exists(realesrgan_path):
            fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hunyuan3D-2.1", "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"))
            realesrgan_path = fallback if os.path.exists(fallback) else realesrgan_path
        conf.realesrgan_ckpt_path = realesrgan_path
        conf.multiview_cfg_path = os.path.join(paint_dir, "cfgs", "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(paint_dir, "hunyuanpaintpbr")
        conf.bake_exp = getattr(settings, "hunyuan_paint_bake_exp", 4)
        ref_bg = getattr(settings, "hunyuan_paint_reference_bg", "255,255,255")
        try:
            conf.reference_bg_color = tuple(int(x.strip()) for x in ref_bg.split(","))[:3]
        except (ValueError, AttributeError):
            conf.reference_bg_color = (255, 255, 255)
        app.state.hunyuan_paint = Hunyuan3DPaintPipeline(conf)
        logger.info("Hunyuan paint pipeline loaded")
    except Exception as e:
        logger.exception(f"Hunyuan paint loading: {e}")
        raise SystemExit("Hunyuan paint failed to load")

    try:
        logger.info("Loading Qwen Edit model...")
        torch.cuda.empty_cache()
        gc.collect()
        app.state.qwen_edit = QwenEditModule(settings)
        await app.state.qwen_edit.startup()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Qwen model loaded")
    except Exception as e:
        logger.exception(f"Qwen model loading: {e}")
        raise SystemExit("Qwen model failed to load")

    save_dir = _hunyuan_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    app.state.hunyuan_save_dir = save_dir

    logger.info("Warming up pipeline...")
    try:
        warmup_image_path = os.path.join(os.path.dirname(__file__), "warmup_image.png")
        if os.path.isfile(warmup_image_path):
            warmup_image = Image.open(warmup_image_path)
            _ = generation_block(warmup_image, seed=42)
        clean_vram()
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")
        clean_vram()

    yield

    if app.state.qwen_edit is not None:
        await app.state.qwen_edit.shutdown()


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def _quick_convert_obj_to_glb(obj_path: str, glb_path: str) -> None:
    from modules.hunyuan.paint.convert_utils import create_glb_with_pbr_materials
    base = obj_path.replace(".obj", "")
    textures = {}
    if os.path.isfile(base + ".jpg"):
        textures["albedo"] = base + ".jpg"
    elif os.path.isfile(base + ".png"):
        textures["albedo"] = base + ".png"
    else:
        textures["albedo"] = base + ".jpg"
    if os.path.isfile(base + "_metallic.jpg"):
        textures["metallic"] = base + "_metallic.jpg"
    if os.path.isfile(base + "_roughness.jpg"):
        textures["roughness"] = base + "_roughness.jpg"
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def generation_block(prompt_image: Image.Image, seed: int = -1):
    edit_prompt = "Show this object in three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"

    t_start = time()

    logger.info("Editing image with Qwen...")
    edited_image = app.state.qwen_edit.edit_image(
        prompt_image=prompt_image,
        seed=seed,
        prompt=edit_prompt,
    )
    logger.debug(f"Qwen editing took: {time() - t_start:.2f}s")

    logger.info("Rembg (BiRefNet) + crop...")
    rgba_image = app.state.birefnet_preprocess(edited_image)

    logger.info("Hunyuan shape generation...")
    seed_use = seed if seed >= 0 else 42
    pipeline = app.state.hunyuan_shape
    device = getattr(pipeline, "device", torch.device("cuda"))
    if seed_use >= 0:
        gen = torch.Generator(device=device).manual_seed(seed_use)
    else:
        gen = None
    mesh = pipeline(
        image=rgba_image,
        num_inference_steps=50,
        guidance_scale=5.0,
        generator=gen,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        num_chunks=8000,
        mc_algo=getattr(settings, "hunyuan_mc_algo", "mc"),
        output_type="trimesh",
        enable_pbar=False,
    )[0]

    save_dir = app.state.hunyuan_save_dir
    initial_glb = os.path.join(save_dir, "current_initial.glb")
    obj_path = os.path.join(save_dir, "current_texturing.obj")
    textured_glb = os.path.join(save_dir, "current_textured.glb")

    mesh.export(initial_glb)
    final_glb_path = initial_glb

    cwd_before = os.getcwd()
    try:
        os.chdir(save_dir)
        try:
            textured_output = app.state.hunyuan_paint(
                mesh_path=initial_glb,
                image_path=rgba_image,
                output_mesh_path=obj_path,
                use_remesh=True,
                save_glb=True,
            )
            if textured_output and textured_output.endswith('.glb') and os.path.exists(textured_output):
                final_glb_path = textured_output
            else:
                # Fallback to manual conversion if pipeline didn't return GLB
                _quick_convert_obj_to_glb(obj_path, textured_glb)
                final_glb_path = textured_glb
        except Exception as e:
            logger.warning(f"Texture generation failed, using untextured mesh: {e}")
            final_glb_path = initial_glb
    finally:
        os.chdir(cwd_before)
    buffer = BytesIO()
    normalize_glb_to_unit_cube(final_glb_path, buffer)
    buffer.seek(0)

    logger.debug(f"Total generation took: {time() - t_start:.2f}s")
    clean_vram()
    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    logger.info("Task received. Prompt-Image")
    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    buffer_size = len(buffer.getvalue())
    buffer.seek(0)
    logger.info("Task completed.")

    async def generate_chunks():
        chunk_size = 1024 * 1024
        while chunk := buffer.read(chunk_size):
            yield chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Length": str(buffer_size)},
    )


@app.get("/version", response_model=str)
async def version() -> str:
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
