# Created 20 Jan 2022 by Andrew Blair-Stanek to make sure that all the
# eventt counts fall within the mathematically reasonable range.

import pickle

print("reading count_onecase_oneevent")
with open("count_onecase_oneevent.pkl", "rb") as f:
    count_onecase_oneevent = pickle.load(f) # count of cases with a chain with e1

print("reading count_onecase_dualevent")
with open("count_onecase_dualevent.pkl", "rb") as f:
    count_onecase_dualevent = pickle.load(f) # count of cases with a chain with e1 and e2

print("reading count_crosscite_oneevent")
with open("count_crosscite_oneevent.pkl", "rb") as f:
    count_crosscite_oneevent = pickle.load(f) # count of c1 cites c2 where both have e1

print("reading count_crosscite_dualevent")
with open("count_crosscite_dualevent.pkl", "rb") as f:
    count_crosscite_dualevent = pickle.load(f) # count of c1 cites c2 where both have e1 and e2

# Check that all one-events fall within the N(N-1) range
idx = 0
for idx, (e1, count_cross_e1) in enumerate(count_crosscite_oneevent.items()):
    if idx % 10000 == 0:
        print("single", idx)
    count_single_e1 = count_onecase_oneevent[e1]
    if count_cross_e1 > 0.5*(count_single_e1*(count_single_e1-1)):
        print("ERROR: ", e1, "count_cross_e1=", count_cross_e1, "count_single_e1=", count_single_e1,
              "total=", 0.5*(count_single_e1*(count_single_e1-1)))

# Check that all dual-events fall within the N(N-1) range
for idx, (e1e2, count_cross_e1e2) in enumerate(count_crosscite_dualevent.items()):
    if idx % 10000 == 0:
        print("double", idx)
    count_single_e1e2 = count_onecase_dualevent[e1e2]
    if count_cross_e1e2 > 0.5*(count_single_e1e2*(count_single_e1e2-1)):
        print("ERROR: ", e1e2, "count_cross_e1e2=", count_cross_e1e2, "count_single_e1e2=", count_single_e1e2,
              "total=", 0.5*(count_single_e1e2*(count_single_e1e2-1)))



