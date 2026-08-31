# -*- coding: utf-8 -*-
"""
Time: 2026/4/23 23:52
Author: lcqin111
Version: V1.0
File: core.py
Description: Public interface module —— Detector class
    Input: model_version (str), img (numpy.ndarray)
    Output: dict containing seven basic expression probabilities, 16 AU predictions, valence and arousal
    EMO_NAMES: ["neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"]
    AU_NAMES: ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7","AU9", "AU10", "AU12", "AU15",
     "AU17", "AU20","AU23", "AU24", "AU25", "AU26"]
"""
import os

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file

from . import reg_mae_3token_cross_attention_v4
from .util.pos_embed import interpolate_pos_embed


_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5135, 0.3905, 0.3437],
                         std=[0.2751, 0.2396, 0.2341]),
])


class Detector:

    def __init__(
        self,
        model_path: str,
        model_name: str = "mae_vit_base_patch16",
        drop_path_rate: float = 0.1,
        device: str = "auto",
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.model = reg_mae_3token_cross_attention_v4.__dict__[model_name](
            norm_pix_loss=False,
            drop_path_rate=drop_path_rate,
        )

        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        checkpoint = load_file(model_path)
        print('loaded model from  "{}"'.format(model_path))
        interpolate_pos_embed(self.model, checkpoint)
        self.model.load_state_dict(checkpoint, strict=True)

    @torch.no_grad()
    def detect(self, img):

        img_tensor = _TRANSFORM(Image.fromarray(img)).unsqueeze(0).to(self.device)
        cls_preds, au_preds, valence_preds, arousal_preds = self.model(img_tensor)

        # Emotion Classification
        cls_probs = F.softmax(cls_preds, dim=1)
        cls_probs_list = [round(x, 4) for x in cls_probs[0].tolist()]

        # AU Classification
        au_probs = torch.sigmoid(au_preds)
        au_probs_list = [round(x, 4) for x in au_probs[0].tolist()]

        # Valence & Arousal
        val = round(torch.clamp(valence_preds, -1.0, 1.0).item(), 4) if valence_preds is not None else None
        aro = round(torch.clamp(arousal_preds, -1.0, 1.0).item(), 4) if arousal_preds is not None else None


        return cls_probs_list, au_probs_list, val, aro

