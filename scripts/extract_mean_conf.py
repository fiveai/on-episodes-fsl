"""
Script to extract the mean + conf interval from models trained on multiple seeds.

SOURCE should be specified as the file with all the numpy saved arrays with the accuracies of the 10.000 epochs for each model

EXPM_ID should be a regex which is unique for the model, where seed should be followed by a '.'.

There are some examples commented out in the code

"""
import numpy as np
import src.utils.meters as meters
import os
import re


def main():
    #SOURCE = "../results/numpy_results/"
    #SOURCE = "../../icml2021-expm-batchsize-matchingnet_numpy/"
    SOURCE = "../../2021-02-02_numpy-results_ox02/"
    #SOURCE = "../../extra_results_for_table/"
    # example EXPM_ID regex to extract results:
    
    #EXPM_ID = ".*resnet12-proto-1shot-CIFARFS-16w-15q-batch256-seed._.*"
    EXPM_ID = ".*resnet12-nca-CIFARFS-batch128-seed._.*"
    #EXPM_ID = ".*resnet12-matching-32w-5s-11q-seed2-CIFARFS-240epochs.*"
    #EXPM_ID = ".*resnet12-proto-32w-5s-11q-seed.-CIFARFS-240epochs.*"
    # seed is directly followed by a ., date of string is replaced by .*, and end of string should finish with .*



    print("EXPM_ID {}".format(EXPM_ID))
    pattern = re.compile(EXPM_ID)

    FILES = []

    for file in os.listdir(SOURCE):
        if pattern.match(file):
            FILES.append(file)

    print(len(FILES))
    print(FILES)

    print("\n>> Validation accuracies")
    print_conf_interval("val", "1", FILES, SOURCE)
    print_conf_interval("val", "5", FILES, SOURCE)

    print("\n>> Test accuracies")
    print_conf_interval("test", "1", FILES, SOURCE)
    print_conf_interval("test", "5", FILES, SOURCE)
    print("\n")

def print_conf_interval(split, shot, ALL_FILES, SOURCE):
    string = split + "_shot" + shot
    FILES = [FILE for FILE in ALL_FILES if string in FILE]
    FULL = np.concatenate([np.load(SOURCE + FILES[i]) for i in range(len(FILES))])
    mean_conf = meters.compute_confidence_interval(FULL)
    print("{} shot = {:.2f}({:.2f})".format(shot, mean_conf[0]*100, mean_conf[1]*100))
    return mean_conf


if __name__ == "__main__":
    main()
