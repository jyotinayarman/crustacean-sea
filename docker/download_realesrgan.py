#!/usr/bin/env python3
import os
import urllib.request

ckpt_path = '/app/modules/hunyuan/paint/ckpt/RealESRGAN_x4plus.pth'
if not os.path.exists(ckpt_path):
    print('Downloading RealESRGAN checkpoint from official GitHub release...')
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    urllib.request.urlretrieve(url, ckpt_path)
    print('RealESRGAN checkpoint downloaded successfully')
else:
    print('RealESRGAN checkpoint already exists')
if os.path.exists(ckpt_path):
    print(f'File size: {os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB')
