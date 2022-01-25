import streamlit as st


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
