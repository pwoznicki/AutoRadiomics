from pathlib import Path

import streamlit as st

from autorad.evaluation import Evaluator, evaluate_feature_dataset
from autorad.inference import infer_utils
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
    run = st_utils.select_run()
    pipeline_artifacts, dataset_artifacts = st_utils.load_artifacts_from_run(
        run
    )
    feature_dataset = infer_utils.load_feature_dataset(
        feature_df=dataset_artifacts["df"],
        dataset_config=dataset_artifacts["dataset_config"],
        splits=dataset_artifacts["splits"],
    )
    result_df = evaluate_feature_dataset(
        dataset=feature_dataset,
        model=pipeline_artifacts["model"],
        preprocessor=pipeline_artifacts["preprocessor"],
    )
    result_df.to_csv(Path(result_dir) / "predictions.csv", index=False)
    # Evaluation
    start = st.button("Evaluate!")
    if start:
        evaluator = Evaluator(
            y_true=result_df["y_true"].tolist(),
            y_pred_proba=result_df["y_pred_proba"].tolist(),
        )
        tab1, tab2, tab3 = st.tabs(
            ["ROC curve", "Precision-recall curve", "Waterfall plot"]
        )
        with tab1:
            fig = evaluator.plot_roc_curve()
            st.plotly_chart(fig)
        with tab2:
            fig = evaluator.plot_precision_recall_curve()
            st.plotly_chart(fig)
        with tab3:
            fig = evaluator.plot_waterfall()
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show()
