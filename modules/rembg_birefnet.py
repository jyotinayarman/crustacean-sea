import numpy as np
import torch
from PIL import Image

from hf_revisions import get_revision


def _get_birefnet_model():
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True, revision=get_revision("ZhengPeng7/BiRefNet")
    )
    model.eval()
    return model


class BiRefNetPreprocess:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_birefnet_model()
            self._model.to(self.device)
        return self._model

    def _rembg(self, image: Image.Image) -> Image.Image:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        out = image.convert("RGBA")
        out.putalpha(mask)
        return out

    def __call__(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        if image.mode == "RGBA":
            alpha = np.array(image)[:, :, 3]
            has_alpha = not np.all(alpha == 255)
        else:
            has_alpha = False
        if max_size and max(image.size) > max_size:
            scale = max_size / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        if not has_alpha:
            image = image.convert("RGB")
            image = self._rembg(image)
        output_np = np.array(image)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if bbox.size == 0:
            return image
        x_min, y_min = np.min(bbox[:, 1]), np.min(bbox[:, 0])
        x_max, y_max = np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size = max(x_max - x_min, y_max - y_min)
        size = int(size * 1)
        x0 = int(center_x - size // 2)
        y0 = int(center_y - size // 2)
        x1 = x0 + size
        y1 = y0 + size
        output_np = output_np[max(0, y0):min(output_np.shape[0], y1), max(0, x0):min(output_np.shape[1], x1)]
        return Image.fromarray(output_np, mode="RGBA")
