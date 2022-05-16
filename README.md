# cross-doc
This is the code accompanying the article Improved Induction of Narrative Chains via Cross-Document Relations (by Andrew Blair-Stanek and Benjamin Van Durme) forthcoming in StarSem 2022.  The raw input data used comes from Harvard Law Library's fantastic Caselaw Access Project (https://case.law), which grants bulk-download access to researchers (the approval process takes a week or two).  Most of this paper's intermediate output and final output is available at an Archive at https://doi.org/10.7281/T1/QVAHMD.  As background, the 965,467 non-trivial federal court decisions (i.e. cases) were broken into 966 shards of at most 1000 cases.  Much of the processing was done shard-by-shard.  Note that data about which cases cite which is available at https://case.law/download/citation_graph/ and was put into a pkl file for speed.  

The co-reference is done by Xia et al. (2020)'s coreference model available at https://github.com/pitrack/incremental-coref/tree/emnlp2020.  (Xia et al. paper is at https://arxiv.org/abs/2005.00128).  

Here is an overview of some key files: 

**extract_events.py** handles extracting event chains for each document (i.e. federal case).  It does this by taking the output from Xia et al.'s coref (in one .jsonlines file per shard) and stitching it with CONLLU output from running Stanford CoreNLP over the text.  Then it uses PredPatt and runs stitching.  The output of event files (one per case for a total of 965k) are available in full at the Archive link above.  

**extract_chain_events.py** takes the event files (which are actually just .txt files) from extract_events.py and, per shard, extracts all the relevant raw statistics used to calculate the five PMI variants discussed in our paper.  

**compile_counts.py** compiles the output of extract_chain_events.py across all shards, thus giving corpus-wide statistics.  

**calc_pmi.py** uses these corpus-wide statistics to calculate the five PMI variants and puts them into PKL files.  (Note that some of the code needs to be un-commented-out to calculate all five).  

**cloze.py** does the actual calculation of cloze statistics using the PKL files for a PMI measure and the TestSplit1 (in the data Archive).  Note that you can set the random.seed, which can slightly alter the statistics by changing which events are omitted to be predicted in the cloze task.  

**agglomerative_cluster.py** does agglomerative clustering using a particular PMI measure.  
