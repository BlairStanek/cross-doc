# Created 27 Jan 2022 by Andrew Blair-Stanek to induce schemas from case.law
# This draws on the pkl'd PMI dicts created by calc_pmi.py.

import os, pickle, time, math, sys
import batch_extract_event_sets
from functools import cmp_to_key

CLUSTER_SIZE = 6 # size at which we stop
exclude_arg = True # excludes events with just "arg"

# moved from calc_pmi_rank.py.
def derive_by_event_dict(d:dict) -> dict:
    rv = dict()
    for (e1, e2), pmi in d.items():
        if e1 not in rv:
            rv[e1] = dict()
        rv[e1][e2] = pmi
        if e2 not in rv:
            rv[e2] = dict()
        rv[e2][e1] = pmi
    return rv


# This assumes that the PMI has already been calculated
# A cluster is defined as a list of events, in sorted order.
def build_cluster_from_seed(seed, pmi_by_e:dict, dynprog_tried:set, ccde:dict, min_count):
    # print("searching for cluster from seed", seed)
    cluster = [seed[0], seed[1]]

    while len(cluster) < CLUSTER_SIZE:
        # First, build a list of candidates that we might add
        candidates = dict()
        for idx1 in range(len(cluster)):
            for idx2 in range(idx1+1, len(cluster)):
                set_intersect = pmi_by_e[cluster[idx1]].keys() & pmi_by_e[cluster[idx2]].keys()
                for candidate in set_intersect:
                    if candidate not in cluster and \
                            (not exclude_arg or candidate[1] != 'arg') and \
                            (ccde.get((cluster[idx1], candidate), 0) >= min_count or \
                             ccde.get((candidate, cluster[idx1]), 0) >= min_count) and \
                            (ccde.get((cluster[idx2], candidate), 0) >= min_count or \
                             ccde.get((candidate, cluster[idx2]), 0) >= min_count):
                        candidates[candidate] = 0.0

        # Now, figure out which of the candidates is best under our metric
        for candidate in candidates.keys():
            for existing in cluster:
                if candidate in pmi_by_e[existing]:
                    candidates[candidate] += pmi_by_e[existing][candidate]

        # Figure out the highest-scoring candidate
        best_e = None
        best_metric = -100.0
        for candidate, metric in candidates.items():
            if metric > best_metric:
                best_e = candidate
                best_metric = metric

        if best_e is not None:
            # print("Best addition is", best_e, "with best_avg_pmi", best_avg_pmi)
            cluster.append(best_e)
            sorted_cluster_str = str(sorted(cluster, key=cmp_to_key(batch_extract_event_sets.compare)))
            if sorted_cluster_str in dynprog_tried:
                print("dynamic programming found duplicate")
                return None, None # we've tried this, not worth trying again
            else:
                dynprog_tried.add(sorted_cluster_str)
        else:
            break # since we didn't get anything, no more cluster to seek out

    # print(cluster)

    # Calculate the total PMI
    total_pmi = 0.0
    for idx1 in range(len(cluster)):
        for idx2 in range(idx1+1, len(cluster)):
            if cluster[idx2] in pmi_by_e[cluster[idx1]]:
                total_pmi += pmi_by_e[cluster[idx1]][cluster[idx2]]
            # add zero for total PMI if they don't appear

    # print(total_pmi, cluster)
    return total_pmi, cluster  # return cluster in order built!


assert len(sys.argv) == 5, "Usage: .py pmi_filename mincount minpmi outfilename"
pmi_filename = sys.argv[1]
min_count = int(sys.argv[2])
min_pmi = float(sys.argv[3])

print("Loading ", pmi_filename)
with open(pmi_filename, "rb") as fpmi:
    pmi_dict = pickle.load(fpmi)
print("Now converting to event by event")
pmi_by_e = derive_by_event_dict(pmi_dict)

# Load the raw counts, and sort them
print("opening count_crosscite_dualevent.pkl")
with open("count_e1e2_CAandCB.pkl", "rb") as f_count:
    ccde = pickle.load(f_count)
print("starting search for seeds")

dynprog_tried = set()
num_seeds_tried = 0
clusters = []
print("Len ccde=", len(ccde))
for idx, (dualevent, count_cc_de) in enumerate(ccde.items()):
    if idx % 10000 == 0:
        print("idx=", idx, "num_seeds_tried=", num_seeds_tried)
    if count_cc_de >= min_count and \
            dualevent[0] in pmi_by_e and \
            pmi_by_e[dualevent[0]].get(dualevent[1],-1) >= min_pmi and \
            (not exclude_arg or (dualevent[0][1] != 'arg' and dualevent[1][1] != 'arg')):
        total_pmi, cluster = build_cluster_from_seed(dualevent, pmi_by_e, dynprog_tried, ccde, min_count)
        if total_pmi is not None:
            clusters.append((total_pmi, cluster))
        num_seeds_tried += 1

print("num_seeds_tried =", num_seeds_tried)
print("len(clusters)=", len(clusters))
print("**********************************************************")

clusters.sort(key=lambda x: x[0])
for total_pmi, cluster in clusters:
    print("{:9.3f}".format(total_pmi), "  ", cluster)
print("writing clusters to", sys.argv[4])
with open(sys.argv[4], "w") as f_out:
    for total_pmi, cluster in clusters:
        cluster_list = sorted(list(cluster))
        f_out.write(str(cluster_list) + "\n")

