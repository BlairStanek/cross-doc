# Created 19 Feb 2022, this gathers all the case ids with a particular
# test split into a specified directory

import sys, pickle, subprocess

testsplit = sys.argv[1]
output_dir = sys.argv[2]

print("testsplit=", testsplit)
print("output_dir=", output_dir)

# load the file about locations of each event files
print("loading file locations...")
FILEPATHS_PKL_FILE = "/export/c07/ablai/dict_caseid_filepath.pkl"
with open(FILEPATHS_PKL_FILE, "rb") as f:
    dict_caseid_filepath = pickle.load(f)

num_caseids = 0
num_caseids_actually_copied = 0
testsplitfile = open(testsplit, "r")
for caseid_line in testsplitfile.readlines():
    if len(caseid_line.strip()) > 0:
        caseid = int(caseid_line.strip())
        num_caseids += 1
        print("caseid=", caseid)
        if caseid in dict_caseid_filepath:
            print("copying")
            filetocopy = dict_caseid_filepath[caseid]
            subprocess.call("cp " + filetocopy + " " + output_dir, shell=True)
            num_caseids_actually_copied += 1

print("num_caseids=", num_caseids)
print("num_caseids_actually_copied=", num_caseids_actually_copied)