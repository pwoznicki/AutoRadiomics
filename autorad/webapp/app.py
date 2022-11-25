import streamlit as st

from autorad.webapp import template_utils


def make_link(page, text):
    return f'<a target="_self" href="http://localhost:8501/{page}">{text}</a>'


def main():
    template_utils.show_title()
    st.markdown(
        f"""
        ### Prerequisites
        Before you start the analysis, you need to specify where your data is located and where you want to store the results.

        Set these in the [Config](http://localhost:8501/Config) :
        - **AUTORAD_INPUT_DIR**: where AutoRadiomics should look for data.
        - **AUTORAD_RESULT_DIR**: where AutoRadiomics should store the immediary and final results.

        ### Quickstart
        To start using AutoRadiomics, you need volumetric imaging data in **DICOM** or **NIfTi** format. 
        
        You will also need segmentations of your regions of interest (ROIs).
        If you don't have them, check out the [Automatic segmentation](http://localhost:8501/Automatic_segmentation/).

        Once you have both images and segmentations, start with:

        #### 1. Feature extraction workflow 🔨
        It consists of three steps:
        - 1.1 [Preparing the dataset](http://localhost:8501/Prepare_dataset/)
        - 1.2 [Data testing and preprocessing (optional)](http://localhost:8501/Preprocess/)
        - 1.3 [Feature extraction](http://localhost:8501/Extract_radiomics_features/)

        Once you have the table with extracted radiomics features, you can start building predictive models for your task.

        #### 2. Binary classification workflow 🧠
        It includes two steps:
        - 2.1 [Training ML models](http://localhost:8501/Train_models/)
        - 2.2 [Model evaluation](http://localhost:8501/Evaluate/)
        
        When you're satisfied with the results, you can use it to make predictions on new data:
        - 3. [Make predictions](http://localhost:8501/Predict/)

        #### Other workflows
        If you're interested in generating volumetric map of radiomics features, check out:
        - [Create radiomics maps](http://localhost:8501/Create_radiomics_maps/)

       

        For more information about AutoRadiomics, please read [our paper](https://www.frontiersin.org/articles/10.3389/fradi.2022.919133/full):
        ```
            AutoRadiomics: A Framework for Reproducible Radiomics Research;
            P Woznicki, F Laqua, T Bley, B Baeßler;
            Frontiers in Radiology, 22
        ```
        Please cite it if you're using the framework for your research.
    """
    )


if __name__ == "__main__":
    main()
