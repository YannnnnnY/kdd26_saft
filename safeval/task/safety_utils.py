import datasets


def filter_beavertails_safe_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if "is_safe" not in dataset.column_names:
        raise KeyError("BeaverTails dataset is missing required column: is_safe")

    return dataset.filter(lambda doc: bool(doc["is_safe"]))
