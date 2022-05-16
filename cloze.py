# Created 16 Feb 2022 to test different PMI measures' ability to do cloze
import sys, random, os, pickle, numpy, gc
from extract_chain_events import event_chains_from_file, compare, twoevents_to_tuple


def calc_recall_at_N(histogram_num_within_N, N, histogram_num_tested, num_tested):
    # calculate Recall@50
    total_within_N = 0
    sanity_check_total_num = 0
    for chain_len in sorted(histogram_num_tested.keys()):
        num = histogram_num_tested[chain_len]
        num_within_N = histogram_num_within_N.get(chain_len, 0)
        print("Recall len {:2d}  {:5d}/{:5d} = {:7.5f}".format(chain_len,
                                                               num_within_N,
                                                               num,
                                                               float(num_within_N)/num))
        total_within_N += num_within_N
        sanity_check_total_num += num
    assert sanity_check_total_num == num_tested
    print("TOTAL recall at", N, "{:5d}/{:5d} = {:7.5f}".format(total_within_N,
                                                      num_tested,
                                                      float(total_within_N)/num_tested))


def calc_stats(num_tested,
               histogram_reciprocal_rank,
               histogram_num_within_50,
               histogram_num_within_5,
               histogram_num_within_1,
               histogram_num_tested):

    calc_recall_at_N(histogram_num_within_50, 50, histogram_num_tested, num_tested)
    calc_recall_at_N(histogram_num_within_5, 5, histogram_num_tested, num_tested)
    calc_recall_at_N(histogram_num_within_1, 1, histogram_num_tested, num_tested)

    # calculate Mean Reciprocal Rank
    sanity_check_total_num = 0
    total_mrr = []
    for chain_len in sorted(histogram_reciprocal_rank.keys()):
        recip_rank = histogram_reciprocal_rank[chain_len]
        print("MRR len {:2d}  num {:5d}   MRR {:9.7f}".format(chain_len,
                                                            len(recip_rank),
                                                            numpy.mean(recip_rank)))
        sanity_check_total_num += len(recip_rank)
        total_mrr.extend(recip_rank)
    assert sanity_check_total_num == num_tested
    print("TOTAL MRR {:5d}  {:9.7f}".format(num_tested,
                                            numpy.mean(total_mrr)))


MIN_CHAIN_LENGTH = 3
MAX_CHAIN_LENGTH = 30

pmi_filename = sys.argv[1] # takes PMI measure as input to command line
shard_dirname = sys.argv[2] # takes the shard with the event files as input

gc.collect()

print("reading pmi dict at", pmi_filename)
with open(pmi_filename, "rb") as f:
    pmi_dict = pickle.load(f)
print("len(pmi_dict)=", len(pmi_dict))
gc.collect()

random.seed(42) # ensures same chain item left out each time for cloze test

# load the sample shard's event chains
num_chains_considered = 0
total_length_of_all_chains_considered = 0
total_length_of_all_chains_without_arg = 0
num_chains_passing_tests = 0
chains = list() # list of 2-tuples, with list-chain (minus missing) and missing as the 2
for filename in os.listdir(shard_dirname):
    if filename.endswith(".events.txt"):
        print("filename=", filename)
        filechains = event_chains_from_file(os.path.join(shard_dirname, filename))
        for chain in filechains:
            num_chains_considered += 1
            total_length_of_all_chains_considered += len(chain)
            chain_without_arg = [] # lists, unlike sets, are guaranteed to maintain order
            for e in chain:
                if e[1] != "arg":
                    chain_without_arg.append(e)
            total_length_of_all_chains_without_arg += len(chain_without_arg)
            if MAX_CHAIN_LENGTH >= len(chain_without_arg) >= MIN_CHAIN_LENGTH:
                num_chains_passing_tests += 1
                # print("raw:", chain_without_arg)
                # chain_list = list(chain_without_arg)

                # Added to ensure complete reproducibility with random index always being the same:
                chain_without_arg.sort()

                idx_to_remove = random.randint(0, len(chain_without_arg)-1) # remove for Cloze
                print(idx_to_remove, end ="\t")
                item_removed = None
                chain_list_minusitem = []
                for idx, item in enumerate(chain_without_arg):
                    if idx_to_remove == idx:
                        item_removed = item
                    else:
                        chain_list_minusitem.append(item)
                chains.append((chain_list_minusitem, item_removed))
        print("len(chains)=", len(chains))

print("Final test set stats:")
print("len(chains)=", len(chains))
print("num_chains_considered=", num_chains_considered)
print("total_length_of_all_chains_considered=", total_length_of_all_chains_considered)
print("total_length_of_all_chains_without_arg =", total_length_of_all_chains_without_arg)
print("num_chains_passing_tests =", num_chains_passing_tests)


# Load the all possible events we might pair with it, so we can loop over them
print("reading count_chain_e1.pkl")
with open("count_chain_e1.pkl", "rb") as f:
    count_chain_e1 = pickle.load(f)

# Now do the actual predictions
num_tested = 0
histogram_num_within_50 = dict() # used to calculcate recall@50
histogram_num_within_5 = dict() # used to calculcate recall@5
histogram_num_within_1 = dict() # used to calculcate recall@1
histogram_reciprocal_rank = dict()
histogram_num_tested = dict()

for chain, missing in chains:
    chain_len = len(chain)

    num_tested += 1
    # print("Chain without Cloze target:", chain)
    candidate_list = []  # list of 2-tuples of event to fill Cloze, plus total PMI
    for idx_e, e in enumerate(count_chain_e1.keys()): # consider all events
        # if idx_e % 10000 == 0:
        #     print(".", end="")
        if e not in chain and e[1] != "arg": # don't consider items already in the chain
                                            # or that are "arg"
            sum_pmi = 0.0
            for e_in in chain: # consider all
                assert compare(e, e_in) != 0, "Should have already checked for not in"
                eventpair = twoevents_to_tuple(e, e_in)
                if eventpair in pmi_dict:
                    sum_pmi += pmi_dict[eventpair]
            if sum_pmi > 0:
                candidate_list.append((e, sum_pmi))
    # print("")
    # sort via sum_pmi
    candidate_list.sort(key=lambda x: x[1], reverse=True)

    # figure out rank of missing item
    print("missing =", missing)
    print("len(candidate_list)=", len(candidate_list))
    # print("Rankings", candidate_list[:20])
    idx_target = None
    for idx, item in enumerate(candidate_list):
        if item[0] == missing:
            idx_target = idx
            break

    histogram_num_tested[chain_len] = 1 + histogram_num_tested.get(chain_len, 0)

    current_reciprocal_rank = None
    if idx_target is None:
        current_reciprocal_rank = 0
    else:
        current_reciprocal_rank = 1/(idx_target+1)
        if idx_target < 50:
            histogram_num_within_50[chain_len] = 1 + histogram_num_within_50.get(chain_len, 0)
        if idx_target < 5:
            histogram_num_within_5[chain_len] = 1 + histogram_num_within_5.get(chain_len, 0)
        if idx_target < 1:
            histogram_num_within_1[chain_len] = 1 + histogram_num_within_1.get(chain_len, 0)

    if chain_len not in histogram_reciprocal_rank:
        histogram_reciprocal_rank[chain_len] = []
    histogram_reciprocal_rank[chain_len].append(current_reciprocal_rank)

    print("Rank", idx_target)
    print("Reciprocal rank=", current_reciprocal_rank)

    print("---------------------")

    if num_tested % 10 == 0: # print the progress
        print("INTERMEDIATE REPORT with num_tested=", num_tested, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        calc_stats(num_tested, histogram_reciprocal_rank,
                   histogram_num_within_50, histogram_num_within_5,  histogram_num_within_1,
                   histogram_num_tested)
        gc.collect()

print("FINAL REPORT")
calc_stats(num_tested, histogram_reciprocal_rank,
           histogram_num_within_50, histogram_num_within_5, histogram_num_within_1,
           histogram_num_tested)
