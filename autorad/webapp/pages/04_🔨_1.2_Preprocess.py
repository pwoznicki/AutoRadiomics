import uuid
from pathlib import Path

import streamlit as st

from autorad.data.dataset import ImageDataset
from autorad.utils import preprocessing, testing
from autorad.webapp import template_utils, utils


def test_in_streamlit(fn):
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
            st.success("Passed!")
            st.write("hey")
        except AssertionError as e:
            st.error(str(e))
        else:
            st.success("Passed!")

    return wrapper


def run_tests(image_dataset: ImageDataset):
    st.write("##### Test your data:")
    n_cases = len(image_dataset.df)
    if st.button("Test if ROI has at least one pixel"):
        test_in_streamlit(
            testing.check_assertion_dataset(
                testing.assert_has_nonzero, image_dataset.mask_paths
            )
        )
    if st.button("Test if image and masks are of the same size"):
        try:
            testing.check_assertion_dataset(
                testing.assert_equal_shape,
                list(zip(image_dataset.image_paths, image_dataset.mask_paths)),
            )
        except AssertionError as e:
            st.error(str(e))
        else:
            st.success(f"Passed for {n_cases} cases!")
    if st.button("Test if ROI in the image has >1 unique values"):
        try:
            testing.check_assertion_dataset(
                testing.assert_has_nonzero_within_roi,
                list(zip(image_dataset.image_paths, image_dataset.mask_paths)),
            )
        except AssertionError as e:
            st.error(str(e))
        else:
            st.success(f"Passed for all {n_cases} cases!")


def show():
    template_utils.show_title()
    st.write(
        "Once you've created the table from previous step, run some basic tests on the images and masks:"
    )
    input_dir = Path(utils.get_input_dir())
    result_dir = Path(utils.get_result_dir())

    image_dataset = template_utils.load_path_df(input_dir=result_dir)
    run_tests(image_dataset)
    template_utils.next_step("1.3_Extract_radiomics_features")
    st.write(
        """
    #####

    Or, if you want to add additional **peritumoral masks** of the region surrounding the segmentation, you can do it below ⬇
    """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        margin = st.number_input(label="Margin (in mm)", value=5)
        margin = int(margin)
    save_dir = input_dir / "peritumoral_masks"
    with col2:
        st.text_input(
            "Where to save the peritumoral masks",
            value=str(save_dir),
        )
    if st.button("Generate peritumoral masks"):
        with st.spinner("Generating border masks..."):
            result_dir.mkdir(exist_ok=True)
            dilated_mask_dir = save_dir / f"border_masks_{margin}"
            dilated_mask_dir.mkdir(exist_ok=True, parents=True)
            extended_paths_df = preprocessing.generate_border_masks(
                dataset=image_dataset,
                margin_in_mm=int(margin),
                output_dir=dilated_mask_dir,
            )
        st.dataframe(extended_paths_df)
        utils.save_table_in_result_dir(
            extended_paths_df,
            "paths_with_border_masks.csv",
            button=False,
        )


if __name__ == "__main__":
    show()
