import streamlit as st


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write(
            """
            Expected format: One folder per class, e.g.
            ```
            train
            +-- dogs
            |   +-- lassie.jpg
            |   +-- komissar-rex.png
            +-- cats
            |   +-- garfield.png
            |   +-- smelly-cat.png
            ```
            """
        )


if __name__ == "__main__":
    show()
