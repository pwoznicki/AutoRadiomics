from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ResampleToMatchd,
)


def load_and_resample_to_match(
    to_resample, reference, interpolation="nearest"
):
    """
    Args:
        to_resample: Path to the image to resample.
        reference: Path to the reference image.
    """
    data_dict = {"to_resample": to_resample, "ref": reference}
    transforms = Compose(
        [
            LoadImaged(("to_resample", "ref")),
            EnsureChannelFirstd(("to_resample", "ref")),
            ResampleToMatchd(
                "to_resample", "ref_meta_dict", mode=interpolation
            ),
        ]
    )
    result = transforms(data_dict)

    return result["to_resample"][0], result["ref"][0]
