# Created 20 Jan 2022 by Andrew Blair-Stanek to count up the number of
# cases and number of citations.  These will be used as denominators to
# determine the probabilities of each combination of event(s), which is
# then used to calculate PMI.

import os, pickle, time

# load the file with info about which cases cite which
print("loading citations...")
CITATIONS_PKL_FILE = "/export/c07/ablai/citations.pkl"
with open(CITATIONS_PKL_FILE, "rb") as f:
    cite_dict = pickle.load(f)

# load the file about locations of each event files
print("loading file locations...")
FILEPATHS_PKL_FILE = "/export/c07/ablai/dict_caseid_filepath.pkl"
with open(FILEPATHS_PKL_FILE, "rb") as f:
    dict_caseid_filepath = pickle.load(f)

# these are the numbers we are calculating
total_num_cases = 0
total_num_cites = 0

EVENTS_DIR = "/export/c07/ablai/Events/"
for shard_dir in os.listdir(EVENTS_DIR):
    print(shard_dir)
    shard_num_cases = 0
    shard_num_cites = 0

    shard_fullpath = os.path.join(EVENTS_DIR, shard_dir)
    assert os.path.isdir(shard_fullpath)
    for filepath in os.listdir(shard_fullpath):
        if filepath.endswith("eventsets.pkl"):
            shard_num_cases += 1
            caseid = int(filepath.strip("caseid_").strip(".eventsets.pkl"))

            # Iterate over cases cited by the current case
            if caseid in cite_dict:  # check that this case cites anything
                for cited_case in cite_dict[caseid]:
                    assert cited_case != caseid, "assume cases should not cite themselves"
                    if cited_case in dict_caseid_filepath:  # this means there is an event to cite
                        assert os.path.exists(dict_caseid_filepath[cited_case]), "assume this file exists"
                        shard_num_cites += 1

    print(" shard_num_cases =", shard_num_cases)
    print(" shard_num_cites =", shard_num_cites)
    total_num_cases += shard_num_cases
    total_num_cites += shard_num_cites

print("total_num_cases =", total_num_cases)
print("total_num_cites =", total_num_cites)


