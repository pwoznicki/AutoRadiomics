from pathlib import Path

import streamlit as st
import utils

from autorad.utils import preprocessing, testing
from autorad.webapp import template_utils


def show():
    st.write(
        "Once you've created the table from previous step, run some basic tests on the images and masks:"
    )
    dataset = template_utils.load_path_df()
    n_cases = len(dataset.df)
    st.write("##### Test your data:")
    if st.button("Test if ROI has at least one pixel"):
        try:
            testing.check_assertion_dataset(
                testing.assert_has_nonzero, dataset.mask_paths
            )
        except AssertionError as e:
            st.error(e)
        else:
            st.success(f"Passed for all {n_cases} cases!")
    if st.button("Test if image and masks are of the same size"):
        try:
            testing.check_assertion_dataset(
                testing.assert_equal_shape,
                list(zip(dataset.image_paths, dataset.mask_paths)),
            )
        except AssertionError as e:
            st.error(e)
        else:
            st.success(f"Passed for {n_cases} cases!")
    if st.button("Test if ROI in the image has >1 unique values"):
        try:
            testing.check_assertion_dataset(
                testing.assert_has_nonzero_within_roi,
                list(zip(dataset.image_paths, dataset.mask_paths)),
            )
        except AssertionError as e:
            st.error(e)
        else:
            st.success(f"Passed for all {n_cases} cases!")
    st.write(
        """
    #####
    You can proceed to the next step, [**Feature extraction**](#feature-extraction). \n
    Or, if you want to add additional **peritumoral masks** of the region surrounding the segmentation, you can do it below ⬇
    """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        margin = st.number_input(label="Margin (in mm)", value=5)
        margin = int(margin)
    input_dir = Path(utils.get_input_dir())
    result_dir = Path(utils.get_result_dir())
    save_dir = input_dir / "peritumoral_masks"
    with col2:
        st.text_input(
            "Where to save the peritumoral masks",
            value=str(save_dir),
        )
    if "gen_started" not in st.session_state:
        st.session_state["get_started"] = False
    if st.button("Generate peritumoral masks"):
        st.session_state["gen_started"] = True
    if st.session_state["gen_started"]:
        with st.spinner("Generating border masks..."):
            result_dir.mkdir(exist_ok=True)
            dilated_mask_dir = save_dir / f"border_masks_{margin}"
            dilated_mask_dir.mkdir(exist_ok=True, parents=True)
            extended_paths_df = preprocessing.generate_border_masks(
                dataset=dataset,
                margin_in_mm=int(margin),
                output_dir=dilated_mask_dir,
            )
        st.dataframe(extended_paths_df)
        utils.save_table_in_result_dir(
            extended_paths_df, "paths_with_border_masks.csv"
        )


if __name__ == "__main__":
    show()
