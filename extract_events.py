# Created 27 Sep 2021 by Andrew Blair-Stanek to extract events from case.law
# files by using the long-coref documents and PredPatt. Borrows from Rachel's munging-scripts
# # merge_coref_predpatt.

import stitch
import json, os, codecs, sys
from predpatt import load_conllu, PredPattOpts, PredPatt
from predpatt.util.ud import dep_v1, dep_v2

CONLLU_LEMMA_IDX = 2
CONLLU_POS_IDX = 4
CONLLU_GOVERNOR_IDX = 6
CONLLU_DEPTYPE_IDX = 7

# largely adapted from conllu2predpatt.py
def get_pred_arg_pairs(ppatt):
    pred_arg_pairs = []
    for pred in ppatt.events:
        pred_head_idx = pred.root.position
        for argument in pred.arguments:
            argument_head_idx = argument.root.position
            pred_arg_pairs.append((pred_head_idx,argument_head_idx))
    return pred_arg_pairs

# Unfortunately, the PredPatt load_conllu function does not load the POS or the
# lemma.  (They are loaded into a _).  Rachel's munging-scripts worked around this
# by doing the parse twice, once into conllu and once into a JSON file.  Rather
# than have duplicate files (one conllu, one json), I am simply re-loading it.
# This code is largely adapted from predpatt/utils/load.py.
def load_conllu_full(filename_or_content):
    rv = []
    sent_num = 1
    try:
        if os.path.isfile(filename_or_content):
            with codecs.open(filename_or_content, encoding='utf-8') as f:
                content = f.read().strip()
        else:
            content = filename_or_content.strip()
    except ValueError:
        content = filename_or_content.strip()

    for block in content.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        lines = []
        sent_id = 'sent_%s' % sent_num
        has_sent_id = 0
        for line in block.split('\n'):
            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    sent_id = line[10:].strip()
                    has_sent_id = 1
                else:
                    if not has_sent_id:   # don't take subsequent comments as sent_id
                        sent_id = line[1:].strip()
                continue
            line = line.split('\t') # data appears to use '\t'
            if '-' in line[0]:      # skip multi-tokens, e.g., on Spanish UD bank
                continue
            assert len(line) == 10, line
            lines.append(line)
        rv.append((sent_id, lines))
        sent_num += 1
    return rv

# Largely copied from merge_coref_predpatt.py
def get_dep_path(pred_head_idx:int, arg_head_idx:int, conllu_sentence:list) -> str:
    assert pred_head_idx >= 0
    assert arg_head_idx >= 0
    assert pred_head_idx < len(conllu_sentence)
    assert arg_head_idx < len(conllu_sentence)
    assert not pred_head_idx == arg_head_idx
    path = []
    path_found = False

    # The conllu_sentence sentence indicies are 1-indexed,
    # but the predpatt indices are 0-indexed
    curr_tok = conllu_sentence[arg_head_idx] # travel from arg up to pred
    while not path_found:
        if curr_tok[CONLLU_DEPTYPE_IDX] == "root":
            break
        gov = int(curr_tok[CONLLU_GOVERNOR_IDX])-1 # go to 0-indexed
        if gov == pred_head_idx:
            path_found = True
        # if gov < 0: # if we are at the ROOT
        #     break
        path.append(curr_tok[CONLLU_DEPTYPE_IDX])
        path.append(conllu_sentence[gov][CONLLU_LEMMA_IDX] + "(" + \
                    conllu_sentence[gov][CONLLU_POS_IDX] + ")")
        curr_tok = conllu_sentence[gov]
    if path_found:
        path_name = "->".join(path[::-1])
    else:
        path_name = conllu_sentence[pred_head_idx][CONLLU_LEMMA_IDX]+"->arg"
    return path_name


def extract_events(coref_json:dict,
                   corenlp_sentences:list,
                   corenlp_sentences_full:list,
                   print_debug:bool) -> str: # returns the string of events
    assert len(corenlp_sentences_full) == len(corenlp_sentences)
    rv = ""

    # match up the two versions of the file
    stitches = stitch.stitch_conllu_coref(coref_json, corenlp_sentences, print_debug)

    # run predpatt; this is largely adapted from conllu2predpatt.py
    parses = []
    ppatts = []
    opts = PredPattOpts(simple = True,
                        cut = True,
                        resolve_relcl = True,
                        resolve_amod = True,
                        resolve_appos = True,
                        resolve_poss = True,
                        resolve_conj = True)
    opts.ud = dep_v2.VERSION # This is crucial, since CoreNLP now outputs in UD v.2

    # Do the PredPatt parsing
    for _, parse in corenlp_sentences:
        parses.append(parse)
    for parse in parses:
        # if len(ppatts) == 142:
        #     print("Here!")
        ppatts.append(PredPatt(parse, opts=opts))

    # For debug purposes, print out all the info on the different sentences
    if print_debug:
        for idx_sent, ppatt in enumerate(ppatts):
            print(idx_sent, "****************")
            print("tokens:", [tok for tok in ppatt.tokens ])
            print("POS:", [str(idx_tok) + "/" + tok[4] for idx_tok, tok in enumerate(corenlp_sentences_full[idx_sent][1])])
            print("Raw predpatt:", ppatt)
            pred_arg_pairs = get_pred_arg_pairs(ppatt)
            print("pred_arg_pairs:", pred_arg_pairs)
            print("filtered:", [(p,a) for p, a in pred_arg_pairs if corenlp_sentences_full[idx_sent][1][p][4].startswith("V")])
            print("edges:", ppatt.edges)

    # Run over the coref chains, which are the primary dimension we look at event chains
    corefs = coref_json["predicted_clusters"]
    for chain_idx, chain in enumerate(corefs):
        chain.sort()
        events = []
        coref_tokens = set()
        for mention in chain:
            idx_start_st = 0 # get the index for the start of the mention
            while idx_start_st < len(stitches):
                if stitches[idx_start_st].start_coref_idx == mention[0]: # exact fit
                    break
                elif stitches[idx_start_st].start_coref_idx > mention[0]:
                    idx_start_st -= 1 # back up one; OK if not exact fit; we aim for over-inclusive range
                    break
                else:
                    idx_start_st += 1

            idx_end_st = idx_start_st # get the index for the END of the mention
            while idx_end_st < len(stitches):
                if stitches[idx_end_st].end_coref_idx >= mention[1]:
                    break
                else:
                    idx_end_st += 1

            if print_debug:
                print(idx_start_st, idx_end_st, "text:",
                      [(st.coref_text,  st.start_coref_idx, st.end_coref_idx, st.start_conllu, st.end_conllu) for st in stitches[idx_start_st:idx_end_st+1]])
            coref_tokens.add(" ".join([st.coref_text for st in stitches[idx_start_st:idx_end_st+1]]))

            # We cannot extract events on coref spans that go across multiple conllu sentences, since the
            # relevant event is undefined.
            if idx_start_st < len(stitches) and idx_end_st < len(stitches) and \
                    stitches[idx_start_st].start_conllu[0] == stitches[idx_end_st].end_conllu[0]: # same sentence?
                sent_idx = stitches[idx_start_st].start_conllu[0]
                # print([tok for tok in ppatts[sent_idx].tokens])
                for pred, arg in get_pred_arg_pairs(ppatts[sent_idx]):
                    # print("pred", pred, "arg", arg)
                    if stitches[idx_start_st].start_conllu[1] <= arg <= stitches[idx_end_st].end_conllu[1] and \
                            pred != arg:
                        # extract the relevant dependency
                        events.append(get_dep_path(pred, arg, corenlp_sentences_full[sent_idx][1]))
                        # print("EVENT:", corenlp_sentences_full[sent_idx][1][pred][CONLLU_LEMMA_IDX])
        if len(events) > 1: # just one event does not make a chain
            rv += "## Long-coref chain No. " + str(chain_idx) + " " + str(coref_tokens) + "\n"
            for e in events:
                rv += e + "   "
            rv += "\n"

        # print("\t", end="")
        #     str_mention = "" # not printed out now; used for duplicate detection later
        #     for idx_tok in range(mention[0], mention[1]+1):
        #         print(tokens_all[idx_tok], end= " ")
        #         if tokens_all[idx_tok][:2] == "##":
        #             str_mention += tokens_all[idx_tok][2:]
        #         else:
        #             str_mention += " " + tokens_all[idx_tok]
    return rv

if __name__ == "__main__":
    LongCorefFile = sys.argv[1]
    ConlluDirectory = sys.argv[2]
    EventsOutDirectory = sys.argv[3]
#     SHARD0_LONGCOREF = "/Users/andrew/Desktop/RESEARCH/Schemas/DP_stitch/shard_out0.jsonlines"
    print("Opening long-coref source", LongCorefFile)
    coref_f = open(LongCorefFile, "r")
    for coref_case_text in coref_f.readlines():
        coref_case_json = json.loads(coref_case_text)
        caseid = coref_case_json["sourcecase"]
        # if caseid != "caseid_3831168":  # for debug
        #     continue
        print("processing", caseid)
        # conllu_file = "Shard0_CONLLU/" + caseid + ".txt.conllu"
        conllu_file = ConlluDirectory + "/" + caseid + ".txt.conllu"

        print("Getting conllu from", conllu_file)
        corenlp_sentences = list(load_conllu(conllu_file))
        corenlp_sentences_full = load_conllu_full(conllu_file)

        events_str = extract_events(coref_case_json, corenlp_sentences, corenlp_sentences_full, False)

        events_str = "## caseid = " + caseid + "\n" + events_str

        event_filename = EventsOutDirectory + "/" + caseid + ".events.txt"
        print("writing events to ", event_filename)
        with open(event_filename, "w") as f:
            f.write(events_str)

