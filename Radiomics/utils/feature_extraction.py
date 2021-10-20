import numpy as np
from radiomics import featureextractor
from SimpleITK import GetImageFromArray
from monai.transforms import CropForeground
from collections import defaultdict

# Code used for creating feature maps, probably should be removed

# def extract_features(img_vol, seg_vol):
#     extractor = featureextractor.RadiomicsFeatureExtractor()
#     extractor.settings['normalize']=False

#     for z_coord in seg_vol.shape[2]:
#         seg_slice = seg_vol[:, :, z_coord]
#         if np.sum(seg_slice) > 0:
#             image_slice = img_vol[:, :, z_coord]
#             seg_slice = np.expand_dims(seg_slice, axis=0)
#             # Crop
#             cropper = CropForeground(margin=100, return_coords=True)
#             coords = cropper(seg_slice)
#             x_min, y_min = coords[1][0], coords[1][1]
#             x_max, y_max = coords[2][0], coords[2][1]
#             image_vol = lung_image_vol[x_min:x_max, y_min:y_max, :]

#             xx,yy = np.meshgrid(np.arange(image_slice.shape[1]),np.arange(image_slice.shape[0]))
#             region_labels=np.floor(xx/12)*1024+np.floor(yy/12)
#             prop_imgs = defaultdict(lambda : np.zeros_like(image_slice, dtype=np.float32))
#             out_df_list = []
#             for patch_idx in tqdm(np.unique(region_labels)):
#                 xx_box, yy_box = np.where(region_labels==patch_idx)
#                 c_block = image_slice[xx_box.min():xx_box.max(),
#                                         yy_box.min():yy_box.max()]
#                 c_block = np.expand_dims(c_block, axis=2)
#                 c_label = np.ones_like(c_block).astype(np.uint8)
#                 c_label[0, 0, 0]= 0
#                 out_row = extractor.execute(GetImageFromArray(c_block),
#                                         GetImageFromArray(c_label))
#                 for k,v in out_row.items():
#                     if isinstance(v, (float, np.floating)) or isinstance(v, (np.ndarray, np.generic)):
#                         prop_imgs[k][region_labels == patch_idx] = v
#                 out_df_list += [out_row]

#                 # show the slice and mask
#                 print('Radiomic Images:', len(prop_imgs))
#                 n_axs = m_axs.flatten()
#                 ax1 = n_axs[0]
#                 ax1.imshow(image_slice, cmap = 'gray')
#                 ax1.axis('off')
#                 ax1.set_title('Image')
#                 np.random.seed(2018)
#                 for c_ax, c_prop in zip(n_axs[1:], list(prop_imgs.keys())):
#                     if not 'diagnostics' in c_prop and not 'shape' in c_prop:
#                         map = prop_imgs[c_prop]
#                         maskedmap = np.ma.masked_array(map, mask=(lungmask==0))
#                         c_ax.imshow(image_slice, cmap='gray')
#                         c_ax.imshow(maskedmap, cmap= 'Spectral')
#                         c_ax.axis('off')
#                         c_ax.set_title('{}'.format(c_prop.replace('original_','').replace('_', '\n')))
