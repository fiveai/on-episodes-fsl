import yaml


def load_dataset_yaml(dataset):
    """
    Loads dataset specific variables

    :param dataset (str): dataset name
    :return num-classes (int), dataset directory (str), split directory (str):
    """
    if dataset == "miniimagenet":
        yaml_file = "../dataset_configs/miniimagenet.yaml"
    elif dataset == "tieredimagenet":
        yaml_file = "../dataset_configs/tieredimagenet.yaml"
    elif dataset == "CIFARFS":
        yaml_file = "../dataset_configs/CIFARFS.yaml"
    else:
        raise NotImplementedError("Unknown dataset")

    with open(yaml_file, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_config["num-classes"], yaml_config["data"], yaml_config["split-dir"]
