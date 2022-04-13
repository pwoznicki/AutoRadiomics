from pathlib import Path

import streamlit as st

from classrad.utils.io import load_json

# get the path of the current file with pathlib.Path
json_path = Path(__file__).parent / "pretrained_models.json"
models = load_json(json_path)


def show():
    with st.sidebar:
        modalities = ["CT", "MRI"]
        st.selectbox("Modality", modalities)
        organs = [
            "Brain",
            "Lung",
            "Heart",
            "Liver",
            "Kidney",
            "Spleen",
            "Pancreas",
            "Intestine",
        ]
        st.selectbox("Organ", organs)


if __name__ == "__main__":
    show()
