import streamlit as st

from autorad.evaluation.evaluate import Evaluator
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
    artifacts = infer_utils.get_artifacts_from_best_run()
    result_df = evaluate.evaluate_feature_dataset(
        dataset=feature_dataset,
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"],
    )
    result_df.to_csv(result_dir / "predictions.csv", index=False)
    # Evaluation
    evaluate = st.button("Evaluate!")
    if evaluate:
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
