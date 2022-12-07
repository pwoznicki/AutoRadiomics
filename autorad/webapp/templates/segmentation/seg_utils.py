def get_organ_names(models_metadata: dict):
    organs = []
    for model_meta in models_metadata.values():
        for organ_name in model_meta["labels"].values():
            if organ_name not in organs:
                organs.append(organ_name)
    return sorted(organs)


def get_region_names(models_metadata: dict):
    regions = [model_meta["region"] for model_meta in models_metadata.values()]
    regions = sorted(list(set(regions)))
    return regions


def filter_models_metadata(
    models_metadata: dict, modality: str, region_name: str
):
    matching_models_metadata = {}
    for model_name, model_meta in models_metadata.items():
        if (
            model_meta["modality"] == modality
            and model_meta["region"] == region_name
        ):
            matching_models_metadata[model_name] = model_meta
    return matching_models_metadata


def filter_models_metadata_by_organ(models_metadata: dict, organ: str):
    matching_models_metadata = {}
    for model_name, model_meta in models_metadata.items():
        if organ in model_meta["labels"].values():
            matching_models_metadata[model_name] = model_meta
    return matching_models_metadata


def get_organ_label(model_metadata: dict, organ: str):
    for label in model_metadata["labels"]:
        if model_metadata["labels"][label] == organ:
            return int(label)
