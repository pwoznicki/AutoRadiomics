import streamlit as st
import pandas as pd
import importlib.util
from ruamel.yaml import YAML
from pathlib import Path

# from https://github1s.com/jrieke/traingenerator/blob/HEAD/app/utils.py#L1-L177
def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.

    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file

    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_df(label):
    uploaded_file = st.file_uploader(label)
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)
    return df


def read_yaml(yaml_path):
    """
    Reads .yaml file and returns a dictionary
    """
    yaml_path = Path(yaml_path)
    yaml = YAML(typ="safe")
    data = yaml.load(yaml_path)
    return data


# # from https://github.com/streamlit/streamlit/issues/1019
# def folder_picker(label="Select folder"):
#     root = tk.Tk()
#     root.withdraw()

#     root.wm_attributes("-topmost", 1)

#     # Add button
#     clicked = st.button(label=label)
#     if clicked:
#         dirname = st.text_input(
#             "Selected folder:", filedialog.askdirectory(master=root)
#         )
#     return dirname
