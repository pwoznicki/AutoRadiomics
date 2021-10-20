import os
from os.path import join, exists, isdir
import argparse
import nibabel as nib
from utils import create_maps

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate radiomics maps from image and ROI"
    )
    parser.add_argument(
        "-i",
        "--input",
        help="parent folder containing *images* and *segmentations* folders",
    )
    parser.add_argument("-o", "--output", help="folder to save the generated maps")
    parser.add_argument(
        "-m", "--margin", help="margin in pixel for the map size around ROI"
    )
    args = parser.parse_args()

    input_dir = args.input
    if input_dir is None:
        input_dir = "/data"
    assert isdir(input_dir)

    output_dir = args.output
    if output_dir is None:
        output_dir = join(input_dir, "results")
    image_dir = join(input_dir, "images")
    segmentation_dir = join(input_dir, "segmentations")
    margin = args.margin
    if margin is None:
        margin = 50
    else:
        margin = int(margin)

    all_images = os.listdir(image_dir)
    nii_images = [fname for fname in all_images if fname.endswith(".nii.gz")]

    all_segmentations = os.listdir(segmentation_dir)
    nii_segmentations = [
        fname for fname in all_segmentations if fname.endswith(".nii.gz")
    ]

    if len(nii_images) == 0:
        print("No Nifti images found in the directory!")
    elif len(nii_images) != len(nii_segmentations):
        print("Number of images and segmentations not matching!")
    else:
        for img_id in nii_images:
            if not img_id.endswith("_img.nii.gz"):
                print("False name of the image file:", img_id)
            else:
                img_path = join(image_dir, img_id)
                seg_id = img_id[:-11] + "_seg.nii.gz"
                seg_path = join(segmentation_dir, seg_id)
                if not exists(seg_path):
                    print("Image ", img_id, "has no corresponding segmentation")
                else:
                    img = nib.load(img_path).get_fdata().astype(int)
                    seg = nib.load(seg_path).get_fdata().astype(int)
                    save_dir = join(output_dir, img_id[:-11])
                    create_maps(img, seg, save_dir, margin)
