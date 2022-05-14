# Created 26 Jan 2022 by Andrew Blair-Stanek to get a sense for the PMI data

from collections import Counter
import os, pickle, time, math
import numpy
import scipy.stats

def print_histogram(pmis:dict):
    histogram = Counter()
    for k, v in pmis.items():
        histogram[int(v)] += 1
    hist_sorted = sorted(histogram.items(), key=lambda x: x[0])
    for entry in hist_sorted:
        print(entry)

def print_correlation(pmi_dict1:dict, pmi_dict2:dict):
    vals1 = []
    vals2 = []
    for k, pmi1 in pmi_dict1.items():
        if k in pmi_dict2:
            vals1.append(pmi1)
            vals2.append(pmi_dict2[k])
    assert len(vals1) == len(vals2)
    print("len(pmi_dict1)=", len(pmi_dict1), "len(pmi_dict2)=", len(pmi_dict2), "overlap=", len(vals1))
    print("correlation and pmi:", scipy.stats.pearsonr(vals1, vals2))

print("now reading pmi_single_cross")
with open("pmi_single_cross.pkl", "rb") as f:
    pmi_single_cross = pickle.load(f)
print_histogram(pmi_single_cross)

print("now reading pmi_single_single")
with open("pmi_single_single.pkl", "rb") as f:
    pmi_single_single = pickle.load(f)
print_histogram(pmi_single_single)

print("now reading pmi_dual_within")
with open("pmi_dual_within.pkl", "rb") as f:
    pmi_dual_within = pickle.load(f)
print_histogram(pmi_dual_within)

print("pmi_single_cross and pmi_single_single:")
print_correlation(pmi_single_cross, pmi_single_single)
print("pmi_single_cross and pmi_dual_within:")
print_correlation(pmi_single_cross, pmi_dual_within)
print("pmi_dual_within and pmi_single_single:")
print_correlation(pmi_dual_within, pmi_single_single)

TOP_NUM = 20
print("Top", TOP_NUM, "in pmi_single_cross")
pmi_single_cross_sorted = sorted(pmi_single_cross.items(), key=lambda x: x[1])
for x in pmi_single_cross_sorted[-20:]:
    print(x)

print("Top", TOP_NUM, "in pmi_single_single")
pmi_single_single_sorted = sorted(pmi_single_single.items(), key=lambda x: x[1])
for x in pmi_single_single_sorted[-20:]:
    print(x)

print("Top", TOP_NUM, "in pmi_dual_within")
pmi_dual_within_sorted = sorted(pmi_dual_within.items(), key=lambda x: x[1])
for x in pmi_dual_within_sorted[-20:]:
    print(x)
