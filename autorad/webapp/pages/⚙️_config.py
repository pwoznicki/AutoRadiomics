import os

import streamlit as st
import utils
import validation_utils


def set_env_for_input_and_results():
    input_dir = st.text_input("Set your default data directory:")
    validation_utils.check_if_dir_exists(input_dir, stop_execution=False)
    result_dir = st.text_input("Set your default results directory:")
    validation_utils.check_if_dir_exists(result_dir, stop_execution=False)
    set_env_vars = st.button("Set environment variables")
    if set_env_vars:
        os.environ["AUTORAD_INPUT_DIR"] = input_dir
        os.environ["AUTORAD_RESULT_DIR"] = result_dir
        st.success(f"Set AUTORAD_INPUT_DIR to {input_dir}")
        st.success(f"Set AUTORAD_RESULT_DIR to {result_dir}")
        st.experimental_rerun()


def show():
    st.title("Configure your environment")
    st.subheader("Your current settings:")
    st.write(
        f"**Default data directory**: {utils.get_env_var('AUTORAD_INPUT_DIR')}"
    )
    st.write(
        f"**Default result directory**: {utils.get_env_var('AUTORAD_RESULT_DIR')}"
    )
    st.empty()
    st.empty()
    st.write("#### Edit your settings:")
    set_env_for_input_and_results()


if __name__ == "__main__":
    show()
