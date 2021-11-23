import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st
from ruamel.yaml import YAML
import classrad

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


def upload_file(label):
    uploaded_file = st.file_uploader(label)
    if uploaded_file is None:
        st.stop()
    return uploaded_file


def load_df(label):
    """ """
    uploaded_file = upload_file(label)
    df = pd.read_csv(uploaded_file)
    return df


def load_test_data(out_dir):
    """
    Loads test data to the `out_dir` directory
    """
    test_out_dir = Path(out_dir) / "test_data"
    if not dir_nonempty(test_out_dir):
        st.write("Loading test data to input directory")
        test_data_dir = Path(classrad.__file__).parent / "tests" / "testing_data"
        for fpath in test_data_dir.rglob("*.nii.gz"):
            fpath.copy(test_out_dir)
    # test_data_dir.rglob("*.nrrd").copy(test_out_dir)


def read_yaml(yaml_path):
    """
    Reads .yaml file and returns a dictionary
    """
    yaml_path = Path(yaml_path)
    yaml = YAML(typ="safe")
    data = yaml.load(yaml_path)
    return data


def save_yaml(data, yaml_path):
    """
    Saves a dictionary to .yaml file
    """
    yaml_path = Path(yaml_path)
    yaml = YAML(typ="safe")
    yaml.dump(data, yaml_path)


def dir_nonempty(dir_path):
    path = Path(dir_path)
    if not path.is_dir():
        return False
    has_next = next(path.iterdir(), None)
    if has_next is None:
        return False
    return True


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
