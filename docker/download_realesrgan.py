#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download

ckpt_path = '/app/modules/hunyuan/paint/ckpt/RealESRGAN_x4plus.pth'
if not os.path.exists(ckpt_path):
    print('Downloading RealESRGAN checkpoint...')
    hf_hub_download(
        repo_id='2kpr/Real-ESRGAN',
        filename='RealESRGAN_x4plus.pth',
        local_dir='/app/modules/hunyuan/paint/ckpt',
        local_dir_use_symlinks=False
    )
    print('RealESRGAN checkpoint downloaded successfully')
else:
    print('RealESRGAN checkpoint already exists')
if os.path.exists(ckpt_path):
    print(f'File size: {os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB')
