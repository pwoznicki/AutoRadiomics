import os
from os.path import join, isdir, splitext


if __name__ == '__main__':
    
    image_dir = 
    segmentation_dir = 

    all_images = os.listdir(image_dir)
    nii_images = [fname in all_images if fname.endswith('.nii.gz')]
    
    all_segmentations = os.listdir(seg_dir)
    nii_segmentations = [fname in all_segmentations if fname.endswith('.nii.gz')]

    if len(nii_images) == 0:
        print('No Nifti images found in the directory!')
    elif len(nii_images) != len(nii_segmentations):
        print('Number of images and segmentations not matching!')
    else:
        for img_id in nii_images:
            if not endswith('_img.nii.gz'):
                print('False name of the image file:', img_id)
            else:
                img_path = join(image_dir, img_id)
                seg_id = img_id[:-10] + 'seg.nii.gz'
                seg_path = join(seg_dir, seg_id)
                if not exists(seg_path):
                    print('Image ', img_id, 'has no corresponding segmentation')
                else:
                    img = nib.load(img_path)
                    seg = nib.load(seg_path)


