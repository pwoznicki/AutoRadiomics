import math
import os

import streamlit as st

from autorad.data import ImageDataset
from autorad.inference import infer_utils
from autorad.utils import mlflow_utils
from autorad.visualization import plot_volumes


def get_env_var(name, show_error=False):
    try:
        return os.environ[name]
    except KeyError:
        message = f"{name} not set"
        if show_error:
            st.error(message)
            st.stop()
        return message


def select_run():
    """
    Select model to use for inference.
    """
    st.subheader("Select model")
    start_mlflow = st.button("Browse trained models in MLFlow")
    if start_mlflow:
        mlflow_utils.start_mlflow_server()
        mlflow_utils.open_mlflow_dashboard()

    selection_modes = ["Select best model"]
    mode = st.radio(
        "Do you want to use the best model trained, or select one yourself?",
        selection_modes,
    )
    # TODO: add option to select model yourself
    # if mode_choice == selection_modes[1]:
    #     st.text_input(
    #         "Paste here path to the selected run",
    #         help="Click on `Browse trained models`, open selected run and copy its `Full path`",
    #    )

    if mode == "Select best model":
        run = infer_utils.get_best_run_from_experiment_name("model_training")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return run


def load_artifacts_from_run(run):
    pipeline_artifacts = infer_utils.load_pipeline_artifacts(run)
    dataset_artifacts = infer_utils.load_dataset_artifacts(run)

    return pipeline_artifacts, dataset_artifacts


def show_title():
    st.set_page_config(
        page_title="AutoRadiomics",
        layout="wide",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.title("AutoRadiomics")
    with col2:
        st.write(
            """
        ####
        The easiest framework for experimenting
        using `pyradiomics` and `scikit-learn`.
        """
        )


def show_random_case(dataset: ImageDataset):
    try:
        row = dataset.df.sample(1).iloc[0]
    except IndexError:
        raise IndexError("No cases found")
    st.dataframe(row)
    image_path = row[dataset.image_colname]
    mask_path = row[dataset.mask_colname]
    try:
        fig = plot_volumes.plot_roi(image_path, mask_path)
        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
    except TypeError:
        raise TypeError(
            "Image or mask path is not a string. "
            "Did you correctly set the paths above?"
        )


def code_header(text):
    """
    Insert section header into a jinja file, formatted as Python comment.
    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"


def notebook_header(text):
    """
    Insert section header into a jinja file, formatted as notebook cell.
    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def next_step(title):
    _, _, col = st.columns(3)
    with col:
        next_page = st.button(
            f"Go to the next step ➡️ {title.replace('_', ' ')}"
        )
        if next_page:
            switch_page(title)


# taken from streamlit-extras
def switch_page(page_name: str):
    from streamlit.runtime.scriptrunner import RerunData, RerunException
    from streamlit.source_util import get_pages

    def standardize_name(name: str) -> str:
        return name.lower().replace("_", " ")

    page_name = standardize_name(page_name)

    pages = get_pages(
        "streamlit_app.py"
    )  # OR whatever your main page is called

    for page_hash, config in pages.items():
        if standardize_name(config["page_name"]) == page_name:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name,
                )
            )

    page_names = [
        standardize_name(config["page_name"]) for config in pages.values()
    ]

    raise ValueError(
        f"Could not find page {page_name}. Must be one of {page_names}"
    )
