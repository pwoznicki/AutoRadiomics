from pathlib import Path

import streamlit as st
import template_utils
from jinja2 import Environment, FileSystemLoader

from autorad.utils import io

# get the path of the current file with pathlib.Path
json_path = Path(__file__).parent / "pretrained_models.json"
pretrained_models = io.load_json(json_path)


def show():
    with st.sidebar:
        modalities = ["CT", "MRI"]
        st.selectbox("Modality", modalities)
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
        st.selectbox("Organ", organs)

    # load the template
    env = Environment(loader=FileSystemLoader("webapp/templates/segmentation"))
    template = env.get_template("nnunet_code.py.jinja")
    code = template.render(header=template_utils.code_header, notebook=False)
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

    st.code(code)


if __name__ == "__main__":
    show()
