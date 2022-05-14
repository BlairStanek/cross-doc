# Created by Andrew Blair-Stanek 24 Sep 2021 to stitch together the output of
# Stanford CoreNLP with Patrick Xia's long-coref.

import json
from predpatt import load_conllu

# This is the smallest possible overlap found between coref and conllu
# Often it is a single word, along with the relevant indicies in both sources
class Stitch:
    def __init__(self, coref_text:str, conllu_text:str,
                 start_coref_idx:int, end_coref_idx:int,
                 start_conllu, end_conllu):
        self.coref_text = coref_text
        self.conllu_text = conllu_text
        self.start_coref_idx = start_coref_idx
        self.end_coref_idx = end_coref_idx
        self.start_conllu = start_conllu
        self.end_conllu = end_conllu

def handle_partial_WordPiece(s:str) -> str:
    # handle partial WordPiece tokens
    if s[:2] == "##":
        return s[2:]
    else:
        return s


def stitch_conllu_coref(coref_json:dict, corenlp_sentences:list, print_debug:bool) -> list:
    rv = []

    # Extract from the long-coref file, including tokens
    sents = coref_json["sentences"]
    sent_map = coref_json["sentence_map"]
    coref_tokens = []
    tok_idx = 0
    prev_tok_idx = -1
    for sent in sents:
        for tok_sent_idx, tok in enumerate(sent):
            coref_tokens.append(tok)
            # if tok != "[CLS]" and tok != "[SEP]":
            #     if tok[:2] == "##":
            #         coref_tokens[-1] = (coref_tokens[-1] + tok[2:])
            #     else:
            #         coref_tokens.append(tok)
            if print_debug:
                if sent_map[tok_idx] > prev_tok_idx:
                    print("")  # newline
                print(tok, end=" ")
            prev_tok_idx = sent_map[tok_idx]
            tok_idx += 1
    if print_debug:
        print("\n***************************************************************") # newline

    # Handle the Core NLP file (.conllu)
    dp_tokens = []
    for sent_idx, (_, corenlp_sent) in enumerate(corenlp_sentences):
        for tok_idx, tok in enumerate(corenlp_sent.tokens):
            tok_to_use = tok.replace("\xa0", " ") # turn non-breaking space to regular space
            dp_tokens.append((tok_to_use, (sent_idx, tok_idx)))
            if print_debug:
                print(tok_to_use, end=" ")
        if print_debug:
            print("") # newline

    # There was a bug in the way that I put large paragraphs or sentences into SpanBERT
    # format to be run thru long coref.  Here we are trying to minimize the damage.
    # This also handles all problems with [UNK].
    THRESHOLD_GOOD_MATCHES = 10 # number of good matches before we assume fixed
    good_matches_since_coref_problem = THRESHOLD_GOOD_MATCHES
    idx_coref_problem_start = None
    len_rv_at_first_encounter = None

    # Do the stitching with two advancing indexes, one thru coref tokens, the other
    # through CoreNLP (i.e. dependency parse) tokens
    idx_dp = 0
    idx_coref = 0
    coref_text = ""
    dp_text = ""
    add_coref = True
    add_dp = True
    start_coref = None
    start_dp = None

    while idx_coref < len(coref_tokens) and idx_dp < len(dp_tokens): # Main loop
        coref_problem = False
        if add_coref:
            while idx_coref < len(coref_tokens) and \
                coref_tokens[idx_coref] in ["[UNK]", "[CLS]", "[SEP]"]:
                coref_problem = True
                if print_debug:
                    print("Coref skipping:", coref_tokens[idx_coref])
                idx_coref += 1

            if idx_coref >= len(coref_tokens):
                break # We have run thru all input; nothing more to do; just exit

            # handle partial WordPiece tokens
            token_to_use = handle_partial_WordPiece(coref_tokens[idx_coref])

            coref_text_withspace = coref_text + " " + token_to_use
            coref_text           = coref_text +       token_to_use
            if start_coref is None:
                start_coref = idx_coref
        if add_dp:
            dp_text_withspace = dp_text + " " + dp_tokens[idx_dp][0]
            dp_text           = dp_text +       dp_tokens[idx_dp][0]
            if start_dp is None:
                start_dp = dp_tokens[idx_dp][1]

        store_and_reset = False # if set to True then we save
        if coref_text == dp_text or coref_text_withspace == dp_text or coref_text == dp_text_withspace:
            store_and_reset = True # this causes a lot to be done below
        elif dp_text.startswith(coref_text) or dp_text.startswith(coref_text_withspace):
            add_coref = True
            add_dp = False
            if dp_text.startswith(coref_text_withspace):
                coref_text = coref_text_withspace # use the with-space version
            idx_coref += 1
        elif coref_text.startswith(dp_text) or coref_text.startswith(dp_text_withspace):
            if coref_text.startswith(dp_text_withspace):
                dp_text = dp_text_withspace # use the with-space version
            add_coref = False
            add_dp = True
            idx_dp += 1
        else: # then we have an unusual problem to handle.
            # Handle extra period.  For example, CoreNLP in one instance changed
            # "Wis." to "Wis." and ".".
            if dp_text == "." and len(rv) > 0 and rv[-1].conllu_text[-1] == ".":
                rv[-1].end_conllu = dp_tokens[idx_dp][1] # move index to cover this
                rv[-1].conllu_text += dp_text
                # try the next DP item against the current coref item
                add_dp = True
                idx_dp += 1
                dp_text = ""
                start_dp = None
                add_coref = False # compare new DP text against existing coref text
            elif coref_problem or good_matches_since_coref_problem < THRESHOLD_GOOD_MATCHES:
                # This handles a bug in how some long sentences and/or unusual paragraphs were
                # encoded for SpanBERT for the coref.  Basically, we need to advance the DP index
                # until we have a match with the coref text.  Obviously, we will lose some
                # of the overlapping data, but that happened back in the SpanBERT preprocessing for
                # the coref.  This also handles problems that occur with the [UNK] token.
                if coref_problem: # basically, at the start
                    print("WARNING had coref problem dp_text:", dp_text, " coref_text:", coref_text,
                          "dp context", [x[0] for x in dp_tokens[max(0,idx_dp-5):min(len(dp_tokens),idx_dp+6)]],
                          "coref context", coref_tokens[max(0,idx_coref-5):min(len(coref_tokens), idx_coref+6)])
                    idx_coref_problem_start = idx_coref
                    len_rv_at_first_encounter = len(rv)
                else: # we are doing further fixing
                    print("WARNING had continuing coref problem", good_matches_since_coref_problem)
                    idx_coref = idx_coref_problem_start # return to the original index where problem started!
                    if len(rv) != len_rv_at_first_encounter:
                        print("WARNING: Have to delete ", len(rv) - len_rv_at_first_encounter, " items from rv")
                    del rv[len_rv_at_first_encounter:]  # delete all matches since then, as must be erroneous
                next_coref = handle_partial_WordPiece(coref_tokens[idx_coref])
                assert next_coref != "[SEP]" and next_coref != "[CLS]" and next_coref != "[UNK]"

                # advance along DP until we find a possible match
                idx_dp += 1
                while idx_dp < len(dp_tokens) and \
                      not next_coref.startswith(dp_tokens[idx_dp][0]) and \
                      not dp_tokens[idx_dp][0].startswith(next_coref):
                    if print_debug:
                        print("DP skipping over: ", dp_tokens[idx_dp][0])
                    idx_dp += 1

                if idx_dp >= len(dp_tokens):
                    print("ERROR: Did not figure out had_coref_break problem",
                          "dp context", [x[0] for x in dp_tokens[max(0,idx_dp-5):min(len(dp_tokens),idx_dp+6)]],
                          "coref context", coref_tokens[max(0,idx_coref-5):min(len(coref_tokens), idx_coref+6)])
                    return rv # return what we have and process as much as we can

                # Solution: start over at this new start point where there may be a match
                coref_text = ""
                add_coref = True
                dp_text = ""
                add_dp = True
                start_coref = None # reset counters
                start_dp = None # reset counters
                good_matches_since_coref_problem = 0 # keep track of how far we are from an error
            # MOVED this to a backup to coref_problems; incorrectly matched up an [UNK] * * * to a weird symbol * * *
            elif idx_dp+1 < len(dp_tokens) and idx_coref+1 < len(coref_tokens) and \
                    (dp_tokens[idx_dp+1][0].startswith(coref_tokens[idx_coref+1]) or \
                     coref_tokens[idx_coref+1].startswith(dp_tokens[idx_dp+1][0])) and \
                    coref_tokens[idx_coref] != "[UNK]" and coref_tokens[idx_coref+1] != "[UNK]":
                # then the two are likely the same; just treat them as if they were
                store_and_reset = True # causes lots of happen below
            else:
                print("ERROR: ", coref_text, "  ", dp_text,
                      "dp context", [x[0] for x in dp_tokens[max(0,idx_dp-7):min(len(dp_tokens),idx_dp+6)]],
                      "coref context", coref_tokens[max(0,idx_coref-7):min(len(coref_tokens), idx_coref+6)])
                return rv  # return what we have and process as much as we can
        # End of handling unusual problems

        if store_and_reset: # this is the only code where we add to the list of stitches
            if print_debug:
                print("STITCH: ", coref_text, " <-> ", dp_text)

            rv.append(Stitch(coref_text, dp_text,
                             start_coref_idx=start_coref, end_coref_idx=idx_coref,
                             start_conllu=start_dp,       end_conllu=dp_tokens[idx_dp][1]))
            add_coref = True
            add_dp = True
            idx_coref += 1
            idx_dp += 1
            coref_text = ""
            start_coref = None
            dp_text = ""
            start_dp = None

            # if we had a coref problem, print out how we resolved it
            if good_matches_since_coref_problem < THRESHOLD_GOOD_MATCHES - 1:
                print("  coref: ", rv[-1].coref_text, "\t conllu: ", rv[-1].conllu_text)

            good_matches_since_coref_problem += 1 # we are moving further from any problem

    if idx_coref < len(coref_tokens):
        if not (idx_coref == len(coref_tokens)-1 and coref_tokens[idx_coref] == '[SEP]'):
            print("## WARNING Unused long-coref tokens: ", coref_tokens[idx_coref:])
    if idx_dp < len(dp_tokens):
        print("## WARNING Unused CoreNLP tokens: ", dp_tokens[idx_dp:])

    return rv

if __name__ == "__main__":
    LONGCOREF_FILE = "/Users/andrew/Desktop/RESEARCH/Schemas/core_pp_stitch/shard_out9.jsonlines"
    CONLLU_FILE = "caseid_1890784.txt.conllu"
    CASEID = "caseid_1890784"
    # LONGCOREF_FILE = "case65439.jsonlines"
    # CONLLU_FILE = "statparsed_caseid_65439.conllu"
    coref_json = None
    with open(LONGCOREF_FILE, "r") as coref_f:
        for line in coref_f.readlines():
            coref_json = json.loads(line)
            if coref_json["sourcecase"] == CASEID:
                break

    corenlp_sentences = list(load_conllu(CONLLU_FILE))

    stitches = stitch_conllu_coref(coref_json, corenlp_sentences, True)
    for stitch in stitches:
        print(stitch.coref_text, "<->", stitch.conllu_text,
              "\t\tcoref:", stitch.start_coref_idx, stitch.end_coref_idx,
              "conllu:", stitch.start_conllu, stitch.end_conllu)
