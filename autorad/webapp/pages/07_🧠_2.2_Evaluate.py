import streamlit as st

from autorad.evaluation import Evaluator, evaluate_feature_dataset
from autorad.webapp import st_read, st_utils


def show():
    st_utils.show_title()

    result_dir = st_read.get_result_dir()
    st.write(
        """
        Analyze the results with most widely used metrics such as
        AUC ROC curve, precision-recall curve and confusion matrix.
        """
    )
    artifacts = st_utils.select_model()

    result_df = evaluate_feature_dataset(
        dataset=feature_dataset,
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
    )
    result_df.to_csv(result_dir / "predictions.csv", index=False)
    # Evaluation
    start = st.button("Evaluate!")
    if start:
        evaluator = Evaluator(
            y_true=result_df[label].tolist(),
            y_pred_proba=result_df[pred].tolist(),
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write(evaluator.plot_roc_curve())
        with col2:
            st.write(evaluator.plot_precision_recall_curve())
        fig = evaluator.plot_waterfall()
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show()
