import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import itertools
from typing import Callable, Union, Sequence, Optional, Tuple, List
from SimpleITK import GetImageFromArray
from collections import defaultdict
from radiomics import featureextractor
import logging
logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)  # pyradiomics makes a lot of noise

def equals_2(img):
    return img == 2

#taken from https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

#taken from monai.transforms.utils
def is_positive(img):
    return img > 0

#adapted from monai.transforms.utils
def generate_spatial_bounding_box(
    img: np.ndarray,
    select_fn: Callable = is_positive,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    """
    generate the spatial bounding box of foreground in the image with start-end positions.
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    The bounding boxes edges are aligned with the input image edges.
    This function returns [-1, -1, ...], [-1, -1, ...] if there's no positive intensity.

    Args:
        img: source image to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
    """
    data = img
    data = np.any(select_fn(data), axis=0)
    ndim = len(data.shape)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data.any(axis=ax)
        if not np.any(dt):
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        min_d = max(np.argmax(dt) - margin[di], 0)
        max_d = max(data.shape[di] - max(np.argmax(dt[::-1]) - margin[di], 0), min_d + 1)
        box_start[di], box_end[di] = min_d, max_d

    return box_start, box_end

def create_maps(image_volume, seg_volume, save_dir, margin=50, num_displayed_slices=6):

    expanded_seg_volume = np.expand_dims(seg_volume, axis=0)
    coords_start, coords_end = generate_spatial_bounding_box(img=expanded_seg_volume, margin=[margin, margin, 0])
    print(f'Cropping the image of size {seg_volume.shape} to the region from {coords_start} to {coords_end}')
    
    image_voi = image_volume[coords_start[0]:coords_end[0], \
                coords_start[1]:coords_end[1], coords_start[2]:coords_end[2]] #needs rewriting into zip?
    seg_voi = seg_volume[coords_start[0]:coords_end[0], \
                coords_start[1]:coords_end[1], coords_start[2]:coords_end[2]]
    
    num_slices = image_voi.shape[2]
    selected_slices = [i * (num_slices-1)//(num_displayed_slices+1) for i in range(1, num_displayed_slices+1)]

    feature_classes = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    feature_count = {'firstorder': 19, 'glcm': 24, 'glrlm': 16, 'glszm': 16, 'gldm': 14, 'ngtdm': 5}
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['normalize']=False
    
    for feature_class in feature_classes:
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName(feature_class)
        num_features = feature_count[feature_class]
        
        fig, ax = plt.subplots(num_features+1, num_displayed_slices, figsize = (20, 5*(num_features)))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'{num_features} features from class {feature_class}', fontsize=25)
        print(f'Plotting maps for {num_features} radiomics features')

        for slice_cnt, slice_num in enumerate(selected_slices):
            image_slice = image_voi[:, :, slice_num].T
            seg_slice = seg_voi[:, :, slice_num].T
            xx,yy = np.meshgrid(np.arange(image_slice.shape[1]),np.arange(image_slice.shape[0]))
            region_labels=np.floor(xx/8)*1024+np.floor(yy/8) 
            prop_imgs = defaultdict(lambda : np.zeros_like(image_slice, dtype=np.float32))
            out_df_list = []
            for patch_idx in np.unique(region_labels):
                xx_box, yy_box = np.where(region_labels==patch_idx)
                c_block = image_slice[xx_box.min():xx_box.max(), 
                                        yy_box.min():yy_box.max()]
                if c_block.shape[0] > 1 and c_block.shape[1] > 1:
                    c_block = np.expand_dims(c_block, axis=2)
                    c_block = c_block.astype(np.uint8)
                    c_label = np.ones_like(c_block).astype(np.uint8)
                    c_label[0, 0, 0] = 0
                    out_row = extractor.execute(GetImageFromArray(c_block),
                                            GetImageFromArray(c_label))
                    for k,v in out_row.items():
                        if isinstance(v, (float, np.floating)) or isinstance(v, (np.ndarray, np.generic)):
                            prop_imgs[k][region_labels == patch_idx] = v
                    out_df_list += [out_row]

            ax[0, slice_cnt].imshow(image_slice, cmap = 'gray')
            ax[0, slice_cnt].axis('off')
            ax[0, slice_cnt].set_title(f'Image - slice {slice_num}/{num_slices}')
            cnt = 1
            feature_names = list(prop_imgs.keys())
            feature_names = [name for name in feature_names if not 'diagnostics' in name and not 'shape' in name]
            for feature_name in feature_names:
                map = prop_imgs[feature_name]
                maskedmap = np.ma.masked_array(map, mask=(seg_slice==0))
                ax[cnt, slice_cnt].imshow(image_slice, cmap='gray')
                im = ax[cnt, slice_cnt].imshow(maskedmap, cmap= 'Spectral')
                ax[cnt, slice_cnt].axis('off')
                plt.colorbar(im, ax=ax[cnt, slice_cnt], shrink=0.7, aspect=20*0.7)
                title = feature_name.split('_')[2]
                ax[cnt, slice_cnt].set_title(f'{title} \n slice {slice_num}/{num_slices}')
                cnt += 1
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = join(save_dir, feature_class+'_maps.pdf')
        fig.savefig(save_path)