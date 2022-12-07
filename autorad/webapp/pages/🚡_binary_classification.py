import streamlit as st

from autorad.webapp import template_utils


def main():
    st.set_page_config(
        page_title="AutoRadiomics",
        layout="wide",
    )
    template_utils.show_title()
    template_utils.show_sidebar_and_select_template(
        task="Binary classification"
    )


if __name__ == "__main__":
    main()
