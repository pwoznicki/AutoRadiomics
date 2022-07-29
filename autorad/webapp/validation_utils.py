from pathlib import Path

import streamlit as st


def check_if_dir_exists(dir_path, stop_execution=True):
    if not Path(dir_path).is_dir() or not dir_path:
        st.error(f"The entered path is not a directory ({str(dir_path)})")
        if stop_execution:
            st.stop()
    else:
        st.success(f"The entered path is a directory! ({str(dir_path)})")
