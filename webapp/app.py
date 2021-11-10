import streamlit as st
import mlflow
import collections
import os
from Radiomics.config.config import Config

import utils

# input: result_df
# output: plots


def main():
    # SET UP AVAILABLE TEMPLATES
    template_dict = collections.defaultdict(dict)
    template_dirs = [f for f in os.scandir("templates") if f.is_dir()]
    template_dirs = sorted(template_dirs, key=lambda e: e.name)
    for template_dir in template_dirs:
        try:
            # Templates with task + framework.
            task, framework = template_dir.name.split("_")
            template_dict[task][framework] = template_dir.path
        except ValueError:
            # Templates with task only.
            template_dict[template_dir.name] = template_dir.path

    st.set_page_config(layout="wide")

    col1, col2 = st.columns(2)
    with col1:
        st.title("Radiomics analysis")
    with col2:
        st.write(
            """
        #### Simple pipeline for modelling using `pyradiomics` and `scikit-learn`.
        """
        )
    config = Config()

    with st.sidebar:
        st.write("## Task")
        task = st.selectbox("Select workflow", list(template_dict.keys()))
        if isinstance(template_dict[task], dict):
            framework = st.selectbox(f"Select task", list(template_dict[task].keys()))
            template_dir = template_dict[task][framework]
        else:
            template_dir = template_dict[task]

    # Show template-specific sidebar components (based on sidebar.py in the template dir).
    template_sidebar = utils.import_from_file(
        "template_sidebar", os.path.join(template_dir, "sidebar.py")
    )
    inputs = template_sidebar.show()


if __name__ == "__main__":
    main()
