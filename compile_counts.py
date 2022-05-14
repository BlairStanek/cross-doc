# Created 18 Jan 2022 by Andrew Blair-Stanek to add up the
# counts in counts.pkl throughout the /export/c07/ablai/Events shards.

from collections import Counter
import os, pickle, time, gc, sys

EVENTS_DIR = "/export/c07/ablai/Events/"


count_chain_e1e2 = Counter()
count_chain_e1 = Counter()
count_case_e1e2_AandB = Counter()
count_case_e1e2_A = Counter()
count_case_e1e2_B = Counter()
count_e1e2_CAandCB = Counter()
count_e1_CAandCB = Counter()
count_e1e2_CA = Counter()
count_e1e2_CB = Counter()
count_e1_CA = Counter()
count_e1_CB = Counter()
num_chains = 0
num_cross_cases = 0
num_cross_chains = 0
num_cases_sanitychecking = 0

latest_time = time.perf_counter()

for shard_dir in os.listdir(EVENTS_DIR):
    print(shard_dir)
    counts_filepath = EVENTS_DIR + shard_dir + "/counts.pkl"
    assert os.path.exists(counts_filepath)

    with open(counts_filepath, "rb") as f_counts:
        dict_all = pickle.load(f_counts)

        count_chain_e1e2.update(dict_all["count_chain_e1e2"])
        count_chain_e1.update(dict_all["count_chain_e1"])
        count_case_e1e2_AandB.update(dict_all["count_case_e1e2_AandB"])
        count_case_e1e2_A.update(dict_all["count_case_e1e2_A"])
        count_case_e1e2_B.update(dict_all["count_case_e1e2_B"])
        count_e1e2_CAandCB.update(dict_all["count_e1e2_CAandCB"])
        count_e1_CAandCB.update(dict_all["count_e1_CAandCB"])
        count_e1e2_CA.update(dict_all["count_e1e2_CA"])
        count_e1e2_CB.update(dict_all["count_e1e2_CB"])
        count_e1_CA.update(dict_all["count_e1_CA"])
        count_e1_CB.update(dict_all["count_e1_CB"])

        num_cases_sanitychecking += dict_all["num_cases_sanitychecking"]
        num_chains += dict_all["num_chains"]
        num_cross_cases += dict_all["num_cross_cases"]
        num_cross_chains += dict_all["num_cross_chains"]

    print("   ", num_cases_sanitychecking, num_chains, "    ", time.perf_counter() - latest_time)
    sys.stdout.flush()
    latest_time = time.perf_counter()
    gc.collect() # gets rid of unneeded memory
    print("flushed")
    sys.stdout.flush()


# write out each counter to a separate pickle file (they'll be large)
print("writing count_chain_e1e2")
with open("/export/c07/ablai/count_chain_e1e2.pkl", "wb") as f:
    pickle.dump(count_chain_e1e2, f)

print("writing count_chain_e1")
with open("/export/c07/ablai/count_chain_e1.pkl", "wb") as f:
    pickle.dump(count_chain_e1, f)

print("writing count_case_e1e2_AandB")
with open("/export/c07/ablai/count_case_e1e2_AandB.pkl", "wb") as f:
    pickle.dump(count_case_e1e2_AandB, f)

print("writing count_case_e1e2_A")
with open("/export/c07/ablai/count_case_e1e2_A.pkl", "wb") as f:
    pickle.dump(count_case_e1e2_A, f)

print("writing count_case_e1e2_B")
with open("/export/c07/ablai/count_case_e1e2_B.pkl", "wb") as f:
    pickle.dump(count_case_e1e2_B, f)

print("writing count_e1e2_CAandCB")
with open("/export/c07/ablai/count_e1e2_CAandCB.pkl", "wb") as f:
    pickle.dump(count_e1e2_CAandCB, f)

print("writing count_e1_CAandCB")
with open("/export/c07/ablai/count_e1_CAandCB.pkl", "wb") as f:
    pickle.dump(count_e1_CAandCB, f)

print("writing count_e1e2_CA")
with open("/export/c07/ablai/count_e1e2_CA.pkl", "wb") as f:
    pickle.dump(count_e1e2_CA, f)

print("writing count_e1e2_CB")
with open("/export/c07/ablai/count_e1e2_CB.pkl", "wb") as f:
    pickle.dump(count_e1e2_CB, f)

print("writing count_e1_CA")
with open("/export/c07/ablai/count_e1_CA.pkl", "wb") as f:
    pickle.dump(count_e1_CA, f)

print("writing count_e1_CB")
with open("/export/c07/ablai/count_e1_CB.pkl", "wb") as f:
    pickle.dump(count_e1_CB, f)

print("num_cases_sanitychecking=", num_cases_sanitychecking)
print("num_chains=", num_chains)
print("num_cross_cases=", num_cross_cases)
print("num_cross_chains=", num_cross_chains)
