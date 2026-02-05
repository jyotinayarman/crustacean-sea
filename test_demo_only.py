#!/usr/bin/env python3
"""
Test script to verify Hunyuan3D pipeline works correctly in Amodelsubmit.
Tests shape + paint generation and verifies GLB output.
"""
import sys
import os
import gc
import torch
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply torchvision fix
try:
    from modules.hunyuan.paint.utils.torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

# Setup hy3dshape module alias
import modules.hunyuan.shape as _hy3dshape
sys.modules["hy3dshape"] = _hy3dshape

from modules.hunyuan import Hunyuan3DDiTFlowMatchingPipeline
from modules.hunyuan.paint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from modules.rembg_birefnet import BiRefNetPreprocess
from utils_glb import normalize_glb_to_unit_cube
from config import settings

def test_hunyuan_pipeline():
    print("="*60)
    print("Testing Hunyuan3D Pipeline (Amodelsubmit)")
    print("="*60)
    
    # Use demo image
    image_path = "../Hunyuan3D-2.1/assets/demo.png"
    if not os.path.exists(image_path):
        print(f"Error: Demo image not found: {image_path}")
        return False
    
    print(f"\n[1/4] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    print(f"  Size: {image.size}, Mode: {image.mode}")
    
    device = f"cuda:{settings.hunyuan_gpu}" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Step 1: Background removal
    print("\n[2/4] Background removal (BiRefNet)...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    birefnet = BiRefNetPreprocess(device=device)
    if image.mode == "RGB":
        image = birefnet(image)
        print(f"  After rembg: {image.mode}")
    
    # Step 2: Shape generation
    print("\n[3/4] Shape generation (Hunyuan3D DiT)...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    try:
        pipeline_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            settings.hunyuan_model_path,
            use_safetensors=False
        )
        print("  Shape pipeline loaded")
        
        seed_use = 42
        gen = torch.Generator(device=device).manual_seed(seed_use) if seed_use >= 0 else None
        
        mesh = pipeline_shape(
            image=image,
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=gen,
            box_v=1.01,
            octree_resolution=384,
            mc_level=0.0,
            num_chunks=8000,
            mc_algo="mc"
        )[0]
        
        initial_glb = "test_shape.glb"
        mesh.export(initial_glb)
        print(f"  ✓ Shape saved: {initial_glb}")
    except Exception as e:
        print(f"  ✗ Shape generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Texture generation
    print("\n[4/4] Texture generation (Hunyuan3D Paint)...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    original_cwd = os.getcwd()
    try:
        paint_dir = os.path.join(os.path.dirname(__file__), "modules", "hunyuan", "paint")
        max_num_view = 6
        resolution = 512
        
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        realesrgan_path = os.path.join(paint_dir, "ckpt", "RealESRGAN_x4plus.pth")
        if not os.path.exists(realesrgan_path):
            fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hunyuan3D-2.1", "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"))
            if os.path.exists(fallback):
                realesrgan_path = fallback
            else:
                raise FileNotFoundError(
                    f"RealESRGAN_x4plus.pth not found. Place it at:\n  {realesrgan_path}\n  or\n  {fallback}\n  Download: https://github.com/xinntao/Real-ESRGAN/releases"
                )
        conf.realesrgan_ckpt_path = realesrgan_path
        conf.multiview_cfg_path = os.path.join(paint_dir, "cfgs", "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(paint_dir, "hunyuanpaintpbr")
        
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        print("  Paint pipeline loaded")
        
        # Change to paint cache directory
        try:
            paint_cache_dir = os.path.join(os.path.dirname(__file__), "hunyuan_cache")
            os.makedirs(paint_cache_dir, exist_ok=True)
            os.chdir(paint_cache_dir)
            
            obj_path = os.path.join(original_cwd, "test_textured.obj")
            textured_output = paint_pipeline(
                mesh_path=os.path.join(original_cwd, initial_glb),
                image_path=os.path.join(original_cwd, image_path),
                output_mesh_path=obj_path,
                use_remesh=True,
                save_glb=True,
            )
        finally:
            os.chdir(original_cwd)
        
        # Verify output
        if textured_output and textured_output.endswith('.glb') and os.path.exists(textured_output):
            final_glb = textured_output
        else:
            # Fallback
            final_glb = obj_path.replace('.obj', '.glb')
            if not os.path.exists(final_glb):
                print(f"  ✗ GLB not found: {final_glb}")
                return False
        
        print(f"  ✓ Textured GLB: {final_glb}")
        
        # Normalize and verify
        buffer = BytesIO()
        normalize_glb_to_unit_cube(final_glb, buffer)
        buffer.seek(0)
        glb_size = len(buffer.getvalue())
        print(f"  ✓ Normalized GLB: {glb_size / 1024:.1f} KB")
        
        # Verify GLB is valid
        import trimesh
        scene = trimesh.load(final_glb)
        meshes = list(scene.geometry.values())
        print(f"  ✓ GLB valid: {len(meshes)} mesh(es)")
        if meshes:
            print(f"    Mesh: {len(meshes[0].vertices)} vertices, {len(meshes[0].faces)} faces")
        
        print("\n" + "="*60)
        print("✓ Pipeline test PASSED")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - Shape: {initial_glb}")
        print(f"  - Textured: {final_glb}")
        return True
        
    except Exception as e:
        try:
            os.chdir(original_cwd)
        except NameError:
            pass
        print(f"  ✗ Texture generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hunyuan_pipeline()
    sys.exit(0 if success else 1)
