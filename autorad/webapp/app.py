import streamlit as st


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


if __name__ == "__main__":
    main()
