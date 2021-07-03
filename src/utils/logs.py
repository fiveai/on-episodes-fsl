import getpass
import socket
import os
import json
import shutil
import torch


def log_experiment(repo, datetime_string, aa):
    """
    logs several experiment variables + sha commit to reproduce experiments
    """
    sha = repo.head.object.hexsha
    dict_args = vars(aa)
    dict_args["_commit"] = sha
    dict_args["_datetime"] = datetime_string
    dict_args["_usr"] = getpass.getuser()
    dict_args["_hostname"] = socket.gethostname()
    dict_args["_working_dir"] = os.getcwd()
    print(
        "\n>> Is repo dirty (untracked changes)? "
        + str(repo.is_dirty(untracked_files=True))
    )

    with open("../expm_args/" + datetime_string + aa.expm_id + ".json", "w") as fp:
        json.dump(dict_args, fp, sort_keys=True, indent=4)


def save_checkpoint(
    state, is_best1, is_best5, filename="checkpoint", folder="result/default"
):
    """
    Saves model checkpoint best 1-shot and best 5-shot model
    """
    torch.save(state, folder + "/" + filename + "_checkpoint.pth.tar")
    if is_best1:
        shutil.copyfile(
            folder + "/" + filename + "_checkpoint.pth.tar",
            folder + "/" + filename + "_best1.pth.tar",
        )
    if is_best5:
        shutil.copyfile(
            folder + "/" + filename + "_checkpoint.pth.tar",
            folder + "/" + filename + "_best5.pth.tar",
        )
