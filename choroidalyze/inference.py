import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF
from torchvision import tv_tensors
from torchvision.io import read_image

import os
import numpy as np
from tqdm import tqdm
from pathlib import Path

from choroidalyze.model import UNet
from choroidalyze.metrics import compute_measurement, compute_measure_maps
from choroidalyze.ppole import bscan_utils


class FixShape(T.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (128 - M%128) % 128
        pad_N = (128 - N%128) % 128
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M,N)

    def __repr__(self):
        return self.__class__.__name__



def get_default_img_transforms():
    return T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.5,), std=(0.5,)),
        FixShape()
    ])


class ImgListDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.transform = get_default_img_transforms()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if isinstance(self.img_list[idx], (str, Path)):
            img = read_image(str(self.img_list[idx]))
        elif isinstance(self.img_list[idx], np.ndarray):
            img = torch.from_numpy(self.img_list[idx])
        elif isinstance(self.img_list[idx], torch.Tensor):
            img = img_path_or_object

        img, shape = self.transform(img)
        
        return {'img': img, "crop":shape}


def get_img_list_dataloader(img_list, batch_size=16, num_workers=0, pin_memory=False):
    dataset = ImgListDataset(img_list)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


class Choroidalyzer:
    DEFAULT_MODEL_URL = 'https://github.com/justinengelmann/Choroidalyzer/releases/download/' \
                        'v1.0/choroidalyzer_model_weights.pth'
    DEFAULT_SCALE = (11.49, 3.87)
    DEFAULT_ZSCALE = 125
    DEFAULT_THRESHOLDS = (0.5, 0.5, 0.1)
    DEFAULT_METHOD = 'vertical'

    def __init__(self, model_name='default', device='cpu',
                 default_scale=DEFAULT_SCALE, default_zscale=DEFAULT_ZSCALE, 
                 default_thresholds=DEFAULT_THRESHOLDS, default_method=DEFAULT_METHOD,
                 img_transforms=None, local_weights_path=None):
        self.model_name = model_name
        self.device = device
        self.default_scale = default_scale
        self.default_zscale = default_zscale
        self.default_thresholds = default_thresholds
        self.default_method = default_method
        self.img_transforms = img_transforms or get_default_img_transforms()
        self.local_weights_path = local_weights_path

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
        img, img_shape = self.img_transforms(img)
        return img, img_shape

    @torch.inference_mode()
    def analyze(self, img_path_or_object: [str, Path, torch.Tensor, np.ndarray], thresholds=None, scale=None, method=None) -> dict:
        thresholds = thresholds or self.default_thresholds
        scale = scale or self.default_scale
        method = method or self.default_method

        preds = self.predict(img_path_or_object)

        region_mask = preds[0].ge(thresholds[0])
        vessel_mask = preds[1].ge(thresholds[1])
        region_mask = region_mask.cpu().numpy()
        vessel_mask = vessel_mask.cpu().numpy()

        fov_loc = self.process_fovea_prediction(preds.unsqueeze(0))

        raw_thickness, area, vascular_index, choroid_vessel_area = compute_measurement(reg_mask=region_mask,
                                                                                       vess_mask=vessel_mask,
                                                                                       fovea=fov_loc,
                                                                                       scale=scale, 
                                                                                       method=method)
        thickness = np.mean(raw_thickness)
        return {'thickness': thickness, 'area': area, 'vascular_index': vascular_index,
                'vessel_area': choroid_vessel_area, 'raw_thickness': raw_thickness}

    @torch.inference_mode()
    def analyze_ppole(self, folder_path: [str, Path], ppole_scale=None, threshold=None, method=None, save_visualisations=False) -> dict:
        threshold = threshold or self.default_thresholds[0]
        ppole_scale = ppole_scale or (*self.default_scale, self.default_zscale)
        method = method or self.default_method
        
        # Directory variables
        dirr, fname = os.path.split(folder_path)
        save_path = None
        if save_visualisations:
            save_path = os.path.join(dirr, f'{fname}_output')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        img_list = list(Path(folder_path).glob("bscan*"))
        img_list = sorted(img_list, key=bscan_utils.sort_alphanumeric)
    
        # Inference
        rv_preds, foveas, rvfmasks = self.predict_batch(img_list)
        fovea_slice_num, fovea = bscan_utils.get_fovea(rvfmasks, foveas, self.process_fovea_prediction)
    
        # Run Ppole analysis code
        measure_dict, volmeasure_dict = compute_measure_maps(rvfmasks,
                                                            threshold,
                                                            fovea_slice_num,
                                                            fovea,
                                                            ppole_scale,
                                                            method,
                                                            save_visualisations,
                                                            fname,
                                                            save_path,
                                                            img_list)
        output = {'choroid_thickness_[um]':measure_dict['choroid_thickness'],
                  'choroid_vessel_area_[um2]':measure_dict['choroid_vessel'],
                  'choroid_CVI':measure_dict['choroid_CVI'],
                  'choroid_volume_[mm3]':volmeasure_dict['choroid_thickness'],
                  'choroid_vessel_volume_[mm3]':volmeasure_dict['choroid_vessel']}
    
        return output

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
            out.append(fov_signal.argmax(dim=-1).cpu().numpy())

        return np.array(out).T

    @torch.inference_mode()
    def predict(self, img_path: str):
        img, (M,N) = self._load_transform_image(img_path)
        img = img.to(self.device)
        with torch.no_grad():
            pred = self.model(img.unsqueeze(0)).sigmoid()
        return pred.squeeze(0).cpu()[:,:M,:N]

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



    @torch.inference_mode()
    def _predict_loader(self, loader, thresholds):
        preds = []
        rvfmasks = []
        foveas = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                
                img = batch['img'].to(self.device)
                batch_M, batch_N = batch['crop']
                pred = self.model(img).sigmoid().squeeze()

                pred_map = pred[:,:2].cpu().numpy()
                fovea = self.process_fovea_prediction(pred)

                if isinstance(thresholds, (int, float)):
                    thresholds = np.array([thresholds, thresholds, thresholds])
                map_thresh, fov_thresh = thresholds[:2], thresholds[-1]

                fov_map = pred[:,-1].cpu().numpy()
                ppole = fov_map.max(axis=-1).max(axis=-1) <= fov_thresh
                fovea[ppole] = 0
                
                pred_map = [p >= map_thresh[:,None,None] for p in pred_map]    
                pred_mask = [p[:,:M,:N] for (p, M, N) in zip(pred_map, batch_M, batch_N)]
                pred_prob = [p[:,:M,:N] for (p, M, N) in zip(pred.cpu().numpy(), batch_M, batch_N)]

                rvfmasks.append(pred_prob)
                preds.append(pred_mask)
                foveas.append(fovea)

        rvpreds=np.concatenate(preds).astype(np.int64)
        fpreds=np.concatenate(foveas).astype(np.int64)
        rvfmasks=np.concatenate(rvfmasks)
                    
        return rvpreds, fpreds, rvfmasks
        

    @torch.inference_mode()
    def predict_batch(self, img_list, thresholds=None, batch_size=16, num_workers=0, pin_memory=False):
        if thresholds is None:
            thresholds = np.array(self.default_thresholds)
        loader = get_img_list_dataloader(img_list, 
                                         batch_size=batch_size, 
                                         num_workers=num_workers,
                                         pin_memory=pin_memory)
        output = self._predict_loader(loader, thresholds)
        return output
