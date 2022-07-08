import os
from pathlib import Path

import streamlit as st
import template_utils
from jinja2 import Environment, FileSystemLoader

from autorad.utils import io

# get the path of the current file with pathlib.Path
seg_dir = Path(__file__).parent.parent / "templates/segmentation"
json_path = seg_dir / "pretrained_models.json"
pretrained_models = io.load_json(json_path)


def show():
    with st.sidebar:
        # modalities = ["CT", "MRI"]
        # modality = st.selectbox("Modality", modalities)
        # regions = ["abdomen", "chest", "pelvis", "head"]
        # region = st.selectbox("Region", regions)
        organs = [
            "Brain",
            "Lung",
            "Heart",
            "Liver",
            "Kidney",
            "Spleen",
            "Pancreas",
            "Intestine",
        ]
        st.multiselect("Organs", organs)

    input_dir = st.text_input("Path to the directory with images")
    input_dir = Path(input_dir.strip('"'))
    if not input_dir.is_dir():
        st.error("Directory not found!")
    files = list(input_dir.glob("*.nii.gz"))
    if files:
        st.success(f"{len(files)} images found!")
    model_dim = st.selectbox("Model", ["2D", "3D"])
    model = "2d" if model_dim == "2D" else "3d_fullres"
    # load the template
    task = "Task017_AbdominalOrganSegmentation"
    env = Environment(loader=FileSystemLoader(str(seg_dir)))
    template = env.get_template("nnunet_code.py.jinja")
    code = template.render(
        header=template_utils.code_header,
        notebook=False,
        task=task,
        model=model,
    )
    notebook_code = template.render(
        header=template_utils.notebook_header, notebook=True
    )
    notebook = template_utils.to_notebook(notebook_code)
    # Display donwload/open buttons.
    st.write("")  # add vertical space
    col1, col2 = st.columns(2)
    with col1:
        template_utils.download_button(
            code, "generated-code.py", "🐍 Download (.py)"
        )
    with col2:
        template_utils.download_button(
            notebook, "generated-notebook.ipynb", "📓 Download (.ipynb)"
        )
    run_jupyter = st.button("📂 Open in Jupyter Notebook")
    if run_jupyter:
        with open("generated-notebook.ipynb", "w") as f:
            f.write(notebook)
        os.system("jupyter notebook")
    st.code(code)


if __name__ == "__main__":
    show()
