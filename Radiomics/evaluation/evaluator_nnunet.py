

def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, labels: tuple, **metric_kwargs):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False)
    files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False)
    assert all([i in files_pred for i in files_gt]), "files missing in folder_with_predictions"
    assert all([i in files_gt for i in files_pred]), "files missing in folder_with_gts"
    test_ref_pairs = [(join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred]
    res = aggregate_scores(test_ref_pairs, json_output_file=join(folder_with_predictions, "summary.json"),
                           num_threads=8, labels=labels, **metric_kwargs)
    return res

def nnunet_evaluate_folder():
    import argparse
    parser = argparse.ArgumentParser("Evaluates the segmentations located in the folder pred. Output of this script is "
                                     "a json file. At the very bottom of the json file is going to be a 'mean' "
                                     "entry with averages metrics across all cases")
    parser.add_argument('-ref', required=True, type=str, help="Folder containing the reference segmentations in nifti "
                                                              "format.")
    parser.add_argument('-pred', required=True, type=str, help="Folder containing the predicted segmentations in nifti "
                                                               "format. File names must match between the folders!")
    parser.add_argument('-l', nargs='+', type=int, required=True, help="List of label IDs (integer values) that should "
                                                                       "be evaluated. Best practice is to use all int "
                                                                       "values present in the dataset, so for example "
                                                                       "for LiTS the labels are 0: background, 1: "
                                                                       "liver, 2: tumor. So this argument "
                                                                       "should be -l 1 2. You can if you want also "
                                                                       "evaluate the background label (0) but in "
                                                                       "this case that would not gie any useful "
                                                                       "information.")
    args = parser.parse_args()
    return evaluate_folder(args.ref, args.pred, args.l)