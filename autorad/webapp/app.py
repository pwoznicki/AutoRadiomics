import streamlit as st

from autorad.webapp import utils, webapp_config


def show_sidebar_and_select_template():
    template_dict = webapp_config.TEMPLATE_DICT
    task = "Binary classification"
    with st.sidebar:
        st.write("## Task")
        framework = st.radio("Select task:", list(template_dict[task].keys()))
    template_path = template_dict[task][framework]
    show_template(template_path)


def show_template(template_path):
    template = utils.import_from_file("template", template_path)
    template.show()


def show_title():
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


def main():
    st.set_page_config(
        page_title="AutoRadiomics",
        layout="wide",
    )
    show_title()
    show_sidebar_and_select_template()


if __name__ == "__main__":
    main()
