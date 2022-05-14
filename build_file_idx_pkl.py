# Created 21 Dec 2021 by Andrew Blair-Stanek to create a quick index of where
# all the files are on /export/c07/ablai/Events so that I can track down the
# cited file from each.

import pickle, os

EVENTS_DIR = "/export/c07/ablai/Events"

dict_caseid_filepath = {}

for dirpath in os.listdir(EVENTS_DIR):
    assert os.path.isdir(os.path.join(EVENTS_DIR, dirpath)), "Should all be directories"
    count_files_in_dir = 0
    print(dirpath)
    for filepath in os.listdir(os.path.join(EVENTS_DIR, dirpath)):
        assert not os.path.isdir(filepath), "Should all be files"
        caseid = int(filepath.strip("caseid_").strip(".events.txt"))
        dict_caseid_filepath[caseid] = os.path.join(EVENTS_DIR, dirpath, filepath)
        # print(dict_caseid_filepath[caseid])
        count_files_in_dir += 1
    if count_files_in_dir < 1000:
        print("*******", dirpath, "had only", count_files_in_dir)

pklfile = open("/export/c07/ablai/dict_caseid_filepath.pkl", "wb")
pickle.dump(dict_caseid_filepath, pklfile)
