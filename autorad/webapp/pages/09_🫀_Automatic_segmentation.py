import subprocess
from pathlib import Path

import streamlit as st
from jinja2 import Environment, FileSystemLoader

from autorad.utils import io
from autorad.webapp import segmentation_utils, st_read, st_utils

# get the path of the current file with pathlib.Path
seg_dir = Path(__file__).parent.parent / "templates/segmentation"
json_path = seg_dir / "pretrained_models.json"
models_metadata = io.load_json(json_path)


def show():
    st_utils.show_title()
    input_dir = st_read.get_input_dir()
    model_name = None
    organ_label = None
    with st.sidebar:
        modalities = ["CT", "MRI"]
        modality = st.selectbox("Modality", modalities)
        regions = segmentation_utils.get_region_names(models_metadata)
        region = st.selectbox("Region", regions)
        matching_models_metadata = segmentation_utils.filter_models_metadata(
            models_metadata, modality, region
        )
        organs = segmentation_utils.get_organ_names(matching_models_metadata)
        organ = st.selectbox("Organ", organs)
        if not organ:
            st.warning("No models found for this modality and region.")
        if organ:
            final_models_metadata = (
                segmentation_utils.filter_models_metadata_by_organ(
                    matching_models_metadata, organ
                )
            )
            final_model_names = list(final_models_metadata.keys())
            model_name = st.radio("Available models", final_model_names)
            organ_label = segmentation_utils.get_organ_label(
                final_models_metadata[model_name], organ
            )
    st.markdown(
        """
        ### Instructions
        If you don't have the segmentations for your dataset,
        you have two options: \n
        1. **Manual segmentation** - outline the organ contours by yourself in a program.
        - We recommend using [3D Slicer](https://www.slicer.org/) or [MITK](https://www.mitk.org/wiki/The_Medical_Imaging_Interaction_Toolkit_(MITK)).
        - There may be an extension/functionality (e.g. interpolation between slices) to make the process faster.\n
        2. **Automatic segmentation** - check in the sidebar on the left if there's a trained model available for your use case.
        - Take into account, that it'll only work when the organ and modality are matching.
        - Even then, it's not guaranteed to work well and you should always visually check the results.
        - It **requires a GPU**! The GPU should have at least 4 GB of VRAM.
        - The segmentation uses nnU-Net. If you're using it, please cite the original paper:
        `Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z`
        """
    )
    st.markdown("### Input")
    nifti_dir = segmentation_utils.dicom_to_nifti_expander(data_dir=input_dir)
    st.markdown("### Segmentation with nnU-Net")
    input_dir = st.text_input(
        "Path to the directory with images", value=nifti_dir
    )
    input_dir = Path(input_dir.strip('"'))
    if not input_dir.is_dir():
        st.error("Directory not found!")
    files = list(input_dir.glob("*.nii.gz"))
    if files:
        st.success(f"{len(files)} images found!")
    model_dim = st.selectbox("Model", ["2D", "3D"])
    mode = "2d" if model_dim == "2D" else "3d_fullres"

    # load the template
    if not model_name:
        st.stop()
    st.markdown(
        """
    To create the automatic segmentations, you should run
    the code below in an interactive notebook:
    """
    )
    env = Environment(loader=FileSystemLoader(str(seg_dir)))
    template = env.get_template("nnunet_code.py.jinja")
    model_params = {
        "input_dir": input_dir,
        "model_name": model_name,
        "mode": mode,
        "organ": organ,
        "organ_label": organ_label,
        "modality": modality,
        "region": region,
    }
    code = template.render(
        header=st_read.notebook_header,
        notebook=True,
        **model_params,
    )
    notebook = st_read.to_notebook(code)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        run_jupyter = st.button("📂 Open in Jupyter Notebook")
    with col2:
        st.download_button(
            "📓 Download (.ipynb)",
            notebook,
            "segmentation.ipynb",
        )
    if run_jupyter:
        with open("segmentation.ipynb", "w") as f:
            f.write(notebook)
        subprocess.Popen(["jupyter", "notebook"])
    st.code(code)


if __name__ == "__main__":
    show()
