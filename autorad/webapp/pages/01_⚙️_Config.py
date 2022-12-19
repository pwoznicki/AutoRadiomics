import os

import streamlit as st

from autorad.webapp import st_utils, validation_utils


def set_env_for_input_and_results():
    with st.form("config dirs"):
        input_dir = st.text_input("Set your default data directory:")
        result_dir = st.text_input("Set your default results directory:")
        set_env_vars = st.form_submit_button("Set environment variables")
        if set_env_vars:
            validation_utils.check_if_dir_exists(
                input_dir, stop_execution=False
            )
            validation_utils.check_if_dir_exists(
                result_dir, stop_execution=False
            )
            os.environ["AUTORAD_INPUT_DIR"] = input_dir
            os.environ["AUTORAD_RESULT_DIR"] = result_dir
            st.success(f"Set AUTORAD_INPUT_DIR to {input_dir}")
            st.success(f"Set AUTORAD_RESULT_DIR to {result_dir}")
            st.experimental_rerun()


def show():
    input_dir = st_utils.get_env_var("AUTORAD_INPUT_DIR")
    result_dir = st_utils.get_env_var("AUTORAD_RESULT_DIR")

    st.title("Configure your environment")
    st.subheader("Your current settings:")
    st.write(f"**Data directory**: `{input_dir}`")
    validation_utils.check_if_dir_exists(input_dir, stop_execution=False)
    st.write(f"**Result directory**: `{result_dir}`")
    validation_utils.check_if_dir_exists(result_dir, stop_execution=False)
    st.empty()
    st.empty()
    st.write("#### Edit your settings:")
    set_env_for_input_and_results()


if __name__ == "__main__":
    show()
