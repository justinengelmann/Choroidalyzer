import torch
from torch import nn
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torchvision.io import read_image

import numpy as np
from tqdm import tqdm
from pathlib import Path

from choroidalyze.model import UNet
from choroidalyze.metrics import compute_measurement


def get_default_img_transforms():
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,)),
    ])


class Choroidalyzer:
    DEFAULT_MODEL_URL = 'https://github.com/justinengelmann/Choroidalyzer/releases/download/' \
                        'v1.0/choroidalyzer_model_weights.pth'
    DEFAULT_SCALE = (11.49, 3.87)
    DEFAULT_THRESHOLDS = (0.5, 0.5, 0.1)

    def __init__(self, model_name='default', device='cpu',
                 default_scale=DEFAULT_SCALE, default_thresholds=DEFAULT_THRESHOLDS,
                 img_transforms=None, local_weights_path=None,
                 override_fovea_to_center=False, macula_rum=3000):
        self.model_name = model_name
        self.device = device
        self.default_scale = default_scale
        self.default_thresholds = default_thresholds
        self.img_transforms = img_transforms or get_default_img_transforms()
        self.local_weights_path = local_weights_path
        self.override_fovea_to_center = override_fovea_to_center
        self.macula_rum = macula_rum

        self._init_model()
        self.outputs = ['region', 'vessel', 'fovea']
        self.fovea_signal_filter = None

    def _init_model(self):
        assert self.model_name == 'default', 'Only default model is supported at this time'

        self.model = UNet(in_channels=1, out_channels=3, depth=7, channels='8_doublemax-64',
                          up_type='conv_then_interpolate', extra_out_conv=True)
        if self.local_weights_path:
            state_dict = torch.load(self.local_weights_path, map_location='cpu')
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.DEFAULT_MODEL_URL, map_location='cpu')

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _load_image_if_needed(self, img_path_or_object: [str, Path, torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(img_path_or_object, (str, Path)):
            img = read_image(str(img_path_or_object))
        elif isinstance(img_path_or_object, np.ndarray):
            img = torch.from_numpy(img_path_or_object)
        elif isinstance(img_path_or_object, torch.Tensor):
            img = img_path_or_object
        else:
            raise ValueError(f'Invalid input type for loading image. Must be str, Path, torch.Tensor, or np.ndarray'
                             f', but got {type(img_path_or_object)}. {img_path_or_object}')
        return img

    def _load_transform_image(self, img: str):
        img = self._load_image_if_needed(img)
        img = tv_tensors.Image(img)
        img = self.img_transforms(img)
        return img

    @torch.inference_mode()
    def analyze(self, img_path_or_object: [str, Path, torch.Tensor, np.ndarray], thresholds=None, scale=None) -> dict:
        thresholds = thresholds or self.default_thresholds
        scale = scale or self.default_scale

        preds = self.predict(img_path_or_object)

        region_mask = preds[0].ge(thresholds[0])
        vessel_mask = preds[1].ge(thresholds[1])
        region_mask = region_mask.cpu().numpy()
        vessel_mask = vessel_mask.cpu().numpy()
        
        if not self.override_fovea_to_center:
            fov_loc = self.process_fovea_prediction(preds.unsqueeze(0))
        else:
            fov_loc = None

        try:
            raw_thickness, area, vascular_index, choroid_vessel_area = compute_measurement(reg_mask=region_mask,
                                                                                           vess_mask=vessel_mask,
                                                                                           fovea=fov_loc,
                                                                                           macula_rum=self.macula_rum,
                                                                                           scale=scale)
        except ValueError as e:
            raise ValueError(f'Metrics calculation failed with the following error: {e}\nThis might be due to the fovea detection failing or the region of interest being too large.')
            
        thickness = np.mean(raw_thickness)
        return {'thickness': thickness, 'area': area, 'vascular_index': vascular_index,
                'vessel_area': choroid_vessel_area, 'raw_thickness': raw_thickness}

    @torch.no_grad()
    def process_fovea_prediction(self, preds):
        def _get_fov_filter(kernel_size=21):
            assert kernel_size % 2 == 1
            fov_filter = nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False, padding_mode='reflect', padding='same')
            fov_filter.requires_grad_(False)
            ascending_weights = torch.linspace(0.1, 1, kernel_size // 2)
            fov_filter_weights = torch.cat([ascending_weights, torch.tensor([1.]), ascending_weights.flip(0)])
            fov_filter_weights /= fov_filter_weights.sum()
            fov_filter.weight = torch.nn.Parameter(fov_filter_weights.view(1, 1, -1), requires_grad=False)
            return fov_filter

        def _agg_fov_signal(tens, d=2):
            return tens[:, 2:, :].sum(dim=d)

        if self.fovea_signal_filter is None:
            self.fovea_signal_filter = (_get_fov_filter(kernel_size=21),
                                        _get_fov_filter(kernel_size=51))

        # we need d=2 (vert) and d=3 (horiz)
        out = []
        for d, filter in zip([2, 3], self.fovea_signal_filter):
            filter.to(preds.device)
            fov_signal = _agg_fov_signal(preds, d)
            fov_signal = filter(fov_signal).squeeze()
            out.append(fov_signal.argmax(dim=-1).item())

        return tuple(out)

    @torch.inference_mode()
    def predict(self, img_path: str):
        img = self._load_transform_image(img_path)
        img = img.to(self.device)
        with torch.no_grad():
            pred = self.model(img.unsqueeze(0)).sigmoid()
        return pred.squeeze(0).cpu()

    @torch.inference_mode()
    def predict_and_plot(self, img_path, thresholds=None):
        thresholds = thresholds or self.default_thresholds
        import matplotlib.pyplot as plt
        preds = self.predict(img_path)

        fov_loc = self.process_fovea_prediction(preds.unsqueeze(0))

        if thresholds is not None:
            if isinstance(thresholds, (int, float)):
                thresholds = (thresholds, thresholds, thresholds)
            preds = [_.ge(thresholds[i]) for i, _ in enumerate(preds)]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        img = self._load_image_if_needed(img_path)
        if img.ndim == 2: # add dummy channel dim if needed
            img = img.unsqueeze(0)

        axes[0, 0].imshow(img.permute(1, 2, 0), cmap='gray')
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(preds[0], cmap='gray')
        axes[0, 1].set_title('Region')

        axes[1, 0].imshow(preds[1], cmap='gray')
        axes[1, 0].set_title('Vessel')

        axes[1, 1].imshow(preds[2], cmap='gray')
        axes[1, 1].axvline(fov_loc[0], color='r')
        axes[1, 1].axhline(fov_loc[1], color='r')
        axes[1, 1].set_title('Fovea')

        axes[0, 0].axvline(fov_loc[0], color='r', alpha=0.1)
        axes[0, 0].axhline(fov_loc[1], color='r', alpha=0.1)

        for _ in axes.flatten():
            _.axis('off')

        plt.tight_layout()
        plt.show()
