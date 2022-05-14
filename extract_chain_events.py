# Created 10 Feb 2022.
# Now handles taking output of extract_events.py (one .txt file per case) and
# compiling all possible counts.  Then, compile_counts.py will do the count
# adding up.

import sys, os, pickle, gc
from collections import Counter

def compare(e1, e2) -> int:
    if e1[0] < e2[0]:
        return -1
    elif e1[0] > e2[0]:
        return 1
    else:
        if e1[1] < e2[1]:
            return -1
        elif e1[1] > e2[1]:
            return 1
    return 0

def twoevents_to_tuple(e1, e2): # sorts them to consistent order
    if compare(e1, e2) < 0:
        return (e1, e2)
    return (e2, e1)

def parse_event_chain(line:str) -> set:
    event_chain = set()
    events = line.split()
    for event in events:
        if event.count("->") == 1:
            parent = event.split("->")[0].split("(")[0] # This is where we decided that verb types don't matter
            deptype = event.split("->")[1]
            event_chain.add((parent, deptype))
    return event_chain

def event_chains_from_file(filepath:str) -> list:
    rv = []
    file = open(filepath, "r")
    for l in file.readlines():
        if l[0:2] != "##" and len(l) > 0:
            rv.append(parse_event_chain(l))
    return rv

def dual_events_in_chain(event_chain:set) -> set:
    rv = set()
    for e1 in event_chain:
        for e2 in event_chain:
            if compare(e1, e2) != 0:
                rv.add(twoevents_to_tuple(e1, e2))
    return rv

def dual_events_in_case(event_chains:list) -> set:
    set_dualevents = set()
    for chain in event_chains:
        set_dualevents.update(dual_events_in_chain(chain))
    return set_dualevents


# This takes a shard directory name, runs over all cases within it,
# also handling all cases cited by cases within it, and writes out
# the dictionaries to a counts.pkl file in that shard.
def handle_shard(event_shard_dir:str,
                 cite_dict:dict,
                 dict_caseid_filepath:dict,
                 print_debug:bool,
                 set_leaveout:set):
    # Below are used to calculate pmi_{standard}
    count_chain_e1e2 = Counter()
    count_chain_e1 = Counter()

    # Below are used to calculate pmi_{doc}
    # count of times where A cites B, and both A and B have chain with e1 and e2
    count_case_e1e2_AandB = Counter()
    # count of times where A cites B and A has a chain with e1 and e2 (A is the citing case)
    count_case_e1e2_A = Counter()
    # count of times where A cites B and _B_ has a chain with e1 and e2 (B is the cited case)
    count_case_e1e2_B = Counter()

    # Count of chains C_A and C_B where e1 and e2 in both C_A and C_B; used as numerator in pmi_dual, pmi_cross, and pmi_atom
    count_e1e2_CAandCB = Counter()

    # Count of chains C_A and C_B where e1 in both C_A and C_B; used as denominator in pmi_cross
    count_e1_CAandCB = Counter()

    # Denominators used in pmi_dual
    # Count of chains C_A where e1 and e2 in C_A (done per citation to C_B)
    count_e1e2_CA = Counter()
    # Same but for C_B (i.e. the cited case; done per citation from C_A)
    count_e1e2_CB = Counter()

    # Denominators used in pmi_atom
    # Count of chains C_A where e1 in C_A (done per citation to C_B)
    count_e1_CA = Counter()
    # Count of chains C_B where e1 in C_B (done per citation from C_A)
    count_e1_CB = Counter()

    num_chains = 0 # used solely for pmi_standard
    num_cross_cases = 0
    num_cross_chains = 0
    num_cases_sanitychecking = 0 # used solely for sanity checking

    for case_A_filename in os.listdir(event_shard_dir):
        if case_A_filename.endswith(".events.txt"):
            caseid_A = int(case_A_filename.strip("caseid_").strip(".events.txt"))
            if caseid_A in set_leaveout: # skip over leaveout set (ie, test, dev)
                continue
            num_cases_sanitychecking += 1

            # load case A information
            case_A_filepath = os.path.join(event_shard_dir, case_A_filename)
            assert os.path.isfile(case_A_filepath), "Should be file"
            if print_debug:
                print(case_A_filepath)
            case_A_event_chains = event_chains_from_file(case_A_filepath)

            # calculate the single-chain counts of single and dual events; used *solely* for pmi_{standard}
            for chain in case_A_event_chains:
                num_chains += 1
                count_chain_e1e2.update(dual_events_in_chain(chain))
                count_chain_e1.update(chain)

            # Iterate over cases cited by the current case
            if caseid_A in cite_dict:  # check that this case cites anything
                for caseid_B in cite_dict[caseid_A]:
                    assert caseid_B != caseid_A, "assume cases should not cite themselves"
                    # find where the case that this case cites resides (if we have it)
                    if caseid_B in dict_caseid_filepath and \
                            caseid_B not in set_leaveout: # treat as invisible cited cases in the test or dev set
                        if print_debug:
                            print(".", end="")
                        case_B_filepath = dict_caseid_filepath[caseid_B]
                        case_B_event_chains = event_chains_from_file(case_B_filepath)

                        # do the calculations used for pmi_{doc}
                        case_A_dualevents = dual_events_in_case(case_A_event_chains)
                        case_B_dualevents = dual_events_in_case(case_B_event_chains)
                        num_cross_cases += 1
                        count_case_e1e2_A.update(case_A_dualevents)
                        count_case_e1e2_B.update(case_B_dualevents)
                        count_case_e1e2_AandB.update(case_A_dualevents.intersection(case_B_dualevents))

                        # calculate the per-CHAIN counts
                        for chain_B in case_B_event_chains:
                            for chain_A in case_A_event_chains:
                                num_cross_chains += 1

                                # calculations for pmi_atom denominator
                                count_e1_CA.update(chain_A)
                                count_e1_CB.update(chain_B)

                                intersect = chain_A.intersection(chain_B)
                                # denominator for pmi_cross
                                count_e1_CAandCB.update(intersect)
                                # numerator for pmi_cross, pmi_dual, pmi_atom
                                count_e1e2_CAandCB.update(dual_events_in_chain(intersect))

                                # calculations for pmi_dual denominator
                                count_e1e2_CA.update(dual_events_in_chain(chain_A))
                                count_e1e2_CB.update(dual_events_in_chain(chain_B))

            if print_debug:
                print("")

    pickle_dict = dict()
    pickle_dict["count_chain_e1e2"] = count_chain_e1e2
    pickle_dict["count_chain_e1"] = count_chain_e1
    pickle_dict["count_case_e1e2_AandB"] = count_case_e1e2_AandB
    pickle_dict["count_case_e1e2_A"] = count_case_e1e2_A
    pickle_dict["count_case_e1e2_B"] = count_case_e1e2_B
    pickle_dict["count_e1e2_CAandCB"] = count_e1e2_CAandCB
    pickle_dict["count_e1_CAandCB"] = count_e1_CAandCB
    pickle_dict["count_e1e2_CA"] = count_e1e2_CA
    pickle_dict["count_e1e2_CB"] = count_e1e2_CB
    pickle_dict["count_e1_CA"] = count_e1_CA
    pickle_dict["count_e1_CB"] = count_e1_CB
    pickle_dict["num_chains"] = num_chains
    pickle_dict["num_cross_cases"] = num_cross_cases
    pickle_dict["num_cross_chains"] = num_cross_chains
    pickle_dict["num_cases_sanitychecking"] = num_cases_sanitychecking

    with open(event_shard_dir + "/counts.pkl", "wb") as f_output:
        pickle.dump(pickle_dict, f_output)

    print("wrote ", event_shard_dir + "/counts.pkl")


BATCH_SIZE = 10

SHARD_DIR_PATH = "/export/c07/ablai/Events/shard"

if __name__ == "__main__":
    start_num = int(sys.argv[1])

    # load the caseids to leave out (i.e. test, dev)
    set_leaveout = set()
    leaveout_filenames = ["/export/c07/ablai/caseids_test_split1.txt",
                          "/export/c07/ablai/caseids_test_split2.txt"]
    for filename in leaveout_filenames:
        with open(filename, "r") as f:
            leaveout1 = f.readlines()
        for l in leaveout1:
            set_leaveout.add(int(l.strip()))
        print("len(set_leaveout)=", len(set_leaveout))

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

    for idx in range(start_num, start_num + BATCH_SIZE):
        print("Shard idx=", idx, "+++++++++++++++++++++++++++++++++++++++++++++++++")
        event_shard_dir = SHARD_DIR_PATH + str(idx)
        if os.path.isdir(event_shard_dir):
            print("in event_shard_dir =", event_shard_dir)
            handle_shard(event_shard_dir, cite_dict, dict_caseid_filepath, True, set_leaveout)
        print("flushing gc")
        gc.collect() # gets rid of unneeded memory
        print("done flushing gc")