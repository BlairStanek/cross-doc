import matplotlib.pyplot as plt
import numpy as np


# underlying data
pmi_standard_mrr = [0.0385393, 0.0467328, 0.0431055, 0.0389642, 0.0358136, 0.0324307, 0.0218931, 0.0229124, 0.0124592, 0.013265, 0.0267649, 0.0120807, 0.0246751, 0.0152892437861272]

pmi_dual_mrr = [0.0119661, 0.0112584, 0.0128834, 0.0120058, 0.0126467, 0.010284, 0.0075175, 0.0045146, 0.01303, 0.0129039, 0.008919, 0.0072724, 0.0044386, 0.00602947789017341]

pmi_cross_mrr = [0.0477968, 0.0548867, 0.0550224, 0.057855, 0.0573627, 0.0615954, 0.0524944, 0.0525057, 0.0435978, 0.0436575, 0.0590998, 0.0385931, 0.038711, 0.047265498049133]

pmi_atom_mrr = [0.0349053, 0.0414403, 0.0392721, 0.0345768, 0.0386505, 0.0389022, 0.0301335, 0.0283957, 0.0280411, 0.0305832, 0.0336988, 0.0334443, 0.0231529, 0.0272589550578035]

pmi_doc_mrr = [0.0124278, 0.0126578, 0.0125874, 0.0135238, 0.0104392, 0.0109139, 0.0118705, 0.0057105, 0.0080398, 0.0114645, 0.0069915, 0.0114881, 0.0052019, 0.00732443554913295]

# setup the x axes
ax = plt.gca()
labels = [str(i) for i in range(2, 15)]
labels.append("15+")
ax.set_xticks(np.arange(14))
ax.set_xticklabels(labels)

ax.set_xlabel("Length Test Chain")

ax.set_ylabel("Mean Reciprocal Rank (MRR)")

# actually put in the data
ax.plot(range(14), pmi_standard_mrr, label="$pmi_{standard}$", marker="o")
ax.plot(range(14), pmi_cross_mrr, label="$pmi_{cross}$", marker="s")
ax.plot(range(14), pmi_atom_mrr, label="$pmi_{atom}$", marker="D")
ax.plot(range(14), pmi_dual_mrr, label="$pmi_{dual}$", marker="*")
ax.plot(range(14), pmi_doc_mrr, label="$pmi_{doc}$", marker="o", fillstyle="none")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplots_adjust(right=0.76)

figure = plt.gcf()

figure.set_size_inches(6.5, 2.8)

# plt.show()

plt.savefig('pmi_mrr.eps',
            format='eps',
            dpi=800,
            bbox_inches="tight",
            # orientation="landscape",
            tight_layout=True,
            papertype="letter")

plt.close()