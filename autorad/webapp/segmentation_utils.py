import os
import shutil
from pathlib import Path

import streamlit as st

from autorad.config.type_definitions import PathLike
from autorad.utils import conversion, spatial


def copy_images_to_nnunet(
    input_dir: PathLike,
    output_dir: PathLike,
    in_filename: str = "image.nii.gz",
):
    files = Path(input_dir).rglob(f"*{in_filename}")

    for file in files:
        output_path = Path(output_dir) / f"{file.parent.name}_0000.nii.gz"
        shutil.copy(str(file), str(output_path))


def copy_predictions_from_nnunet(
    pred_dir: PathLike,
    out_dir: PathLike,
    out_filename: str = "segmentation_auto.nii.gz",
):
    files = Path(pred_dir).rglob("*.nii.gz")

    for file in files:
        id_ = file.stem.split("_")[0]
        output_path = Path(out_dir) / id_ / out_filename
        shutil.copy(str(file), str(output_path))


def dicom_to_nifti_expander(data_dir):
    st.write(
        """
    For the later analysis, the images (and the segmentations) need to be
    in the **NIfTI format (.nii.gz)**. \\
    If you have DICOMs, convert them here:
    """
    )
    with st.expander("Convert DICOM to NIFTI"):
        dicom_dir = st.text_input(
            "Choose the root directory with DICOMs:",
            data_dir,
        )
        dicom_dir = Path(dicom_dir)
        if dicom_dir.is_dir():
            subdirs = [d.stem for d in dicom_dir.iterdir() if d.is_dir()]
            st.success(
                f"Directory exists and contains {len(subdirs)} subfolders. Check below if they look good.",
            )
            st.json(subdirs)
        else:
            st.error(
                f"Directory {dicom_dir} does not exist",
            )
        out_dirname = st.text_input("Output directory name:", "nifti")
        out_dir = Path(data_dir) / out_dirname
        out_dir.mkdir(exist_ok=True)
        if len(os.listdir(out_dir)) > 0:
            st.warning(f"Output directory {out_dir} is not empty. ")
        multiple_images = st.checkbox(
            "I have multiple DICOM images per patient (will require manual renaming)",
            value=False,
        )
        if not multiple_images:
            out_filename = "image"
        else:
            out_filename = dicom_dir.name + ".nii.gz"
        run_conversion = st.button("Convert")
        if run_conversion:
            with st.spinner("Converting DICOMs to NIFTI..."):
                conversion.convert_to_nifti(
                    dicom_dir, output_path=out_dir / out_filename
                )
            st.success(f"Done! Nifties saved in {out_dir}")

        return out_dir


def leave_only_organ_segmentation(
    seg_dir: PathLike,
    organ_label: int,
    save_dir: PathLike,
):
    """Takes the segmentations from a trained nnUNet model,
    filters out only the segmentation for `organ_label`, to which label=1 is assigned.
    The other segmentations are cleared (label=0).
    """
    label_map = {organ_label: 1}
    for mask_path in Path(seg_dir).glob("*.nii.gz"):
        save_path = Path(save_dir) / mask_path.name
        spatial.relabel_mask(
            mask_path=mask_path,
            label_map=label_map,
            save_path=save_path,
            strict=False,
            set_rest_to_zero=True,
        )


def get_organ_names(models_metadata: dict):
    organs = []
    for model_meta in models_metadata.values():
        for organ_name in model_meta["labels"].values():
            if organ_name not in organs:
                organs.append(organ_name)
    return sorted(organs)


def get_region_names(models_metadata: dict):
    regions = [model_meta["region"] for model_meta in models_metadata.values()]
    regions = sorted(list(set(regions)))
    return regions


def filter_models_metadata(
    models_metadata: dict, modality: str, region_name: str
):
    matching_models_metadata = {}
    for model_name, model_meta in models_metadata.items():
        if (
            model_meta["modality"] == modality
            and model_meta["region"] == region_name
        ):
            matching_models_metadata[model_name] = model_meta
    return matching_models_metadata


def filter_models_metadata_by_organ(models_metadata: dict, organ: str):
    matching_models_metadata = {}
    for model_name, model_meta in models_metadata.items():
        if organ in model_meta["labels"].values():
            matching_models_metadata[model_name] = model_meta
    return matching_models_metadata


def get_organ_label(model_metadata: dict, organ: str):
    for label in model_metadata["labels"]:
        if model_metadata["labels"][label] == organ:
            return int(label)
