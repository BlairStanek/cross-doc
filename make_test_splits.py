# Creates two lists of federal case IDs that are in non-overlapping
# test splits.  (I.e. one dev, one test).

import random, pickle

PERCENT_TEST1 = 0.003
PERCENT_TEST2 = 0.003

caseids_test_split1 = open("caseids_test_split1.txt", "w")
caseids_test_split2 = open("caseids_test_split2.txt", "w")

with open("../FederalBatchesTest/federal_caseids.pkl", "rb") as f:
    federal_caseids = pickle.load(f)

random.seed(42)
print("len(federal_caseids)=", len(federal_caseids))
for idx, caseid in enumerate(federal_caseids):
    if idx % 10000 == 0:
        print(idx)
    val = random.random()
    if val < PERCENT_TEST1:
        caseids_test_split1.write(str(caseid) + "\n")
    elif val < (PERCENT_TEST1 + PERCENT_TEST2):
        caseids_test_split2.write(str(caseid) + "\n")

caseids_test_split1.close()
caseids_test_split2.close()