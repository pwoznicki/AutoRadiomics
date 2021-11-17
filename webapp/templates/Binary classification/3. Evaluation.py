import streamlit as st
import utils
from Radiomics.evaluation.evaluator import Evaluator
from pathlib import Path

result_dir = Path("/Users/p.woznicki/Documents/test")


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        pass

    # LAYING OUT THE TOP SECTION OF THE APP
    col1, col2 = st.columns((2, 3))

    st.write(
        "##### Analyze the results with most widely used metrics such as AUC ROC curve, precision-recall curve and confusion matrix."
    )
    result_df = utils.load_df("Choose a CSV file with predictions:")
    st.write(result_df)
    result_df_colnames = result_df.columns.tolist()
    label = st.selectbox("Select the label", result_df_colnames)
    # Evaluation
    evaluate = st.button("Evaluate!")
    if evaluate:
        evaluator = Evaluator(
            result_df=result_df,
            target=label,
            result_dir=result_dir,
        )
        evaluator.evaluate()
        st.write(evaluator.plot_roc_curve_all())
        st.write(evaluator.plot_confusion_matrix_all())
        st.write(
            f"""  
            The best performing model in terms of AUC ROC in 5-fold cross-validation is ***{evaluator.best_model_name}**.
            This model is evaluated on the test set:
        """
        )
        st.write(evaluator.plot_test())


if __name__ == "__main__":
    show()
