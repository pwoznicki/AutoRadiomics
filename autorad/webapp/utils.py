import importlib.util
import os
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


def save_table(df, save_path):
    df.to_csv(save_path, index=False)
    st.success(f"Done! Table saved ({save_path})")


def save_table_streamlit(df, save_path, button=True):
    if button:
        saved = st.button("Looks good? Save the table ⬇️")
    else:
        saved = True
    if saved:
        df.to_csv(save_path, index=False)
        st.success(f"Done! Table saved in your result directory ({save_path})")


def get_input_dir():
    if "AUTORAD_INPUT_DIR" in os.environ:
        return os.environ["AUTORAD_INPUT_DIR"]
    else:
        st.error(
            "Oops, input directory not set! Go to the config page and set it."
        )
        st.stop()


def get_result_dir():
    if "AUTORAD_RESULT_DIR" in os.environ:
        return os.environ["AUTORAD_RESULT_DIR"]
    else:
        st.error(
            "Oops, result directory not set! Go to the config page and set it."
        )
        st.stop()


def get_env_var(name, show_error=False):
    try:
        return os.environ[name]
    except KeyError:
        message = f"{name} not set"
        if show_error:
            st.error(message)
            st.stop()
        return message


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, mode="w") as zipf:
        len_dir_path = len(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])


# from https://github1s.com/jrieke/traingenerator
#      /blob/HEAD/app/utils.py#L1-L177
def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.

    Args:
        module_name (str): Assigned to the module's __name__ parameter
        (does not influence how the module is named outside of this function)
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


def dir_nonempty(dir_path):
    path = Path(dir_path)
    if not path.is_dir():
        return False
    has_next = next(path.iterdir(), None)
    if has_next is None:
        return False
    return True
