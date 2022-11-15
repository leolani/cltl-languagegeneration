""" Filename:     thought_utils.py
    Author(s):    Thomas Bellucci, Selene Baez Santamaria
    Description:  Utility functions used by the RLReplier defined in
                  Replier.py.
    Date created: Nov. 11th, 2021
    Date updated: Nov. 9th, 2022
"""

import random
from itertools import combinations


def thoughts_from_brain(utt, cap, filter=None):
    """Takes a brain response capsule and extracts thoughts from it in the form of
    a dictionary, e.g. {'object_gap person book':('_object_gap', thought_dict), ...}.
    The information in the keys is used by the RLReplier to make a decision.

    params
    dict capsule: dict containing the input utterance, triples, perspectives
                  and contextual information (e.g. location, speaker)
    object typer: Typing object that maps a token to a type (a hypernym).

    returns:      dict mapping from thought names to (thought_type, thought_info)
    """

    # Trust is always available
    thoughts = dict()
    if "_trust" in filter:
        thoughts["_trust"] = ("trust", cap["_trust"])

    if "_statement_novelty" in filter:
        # Any statement novelties? (can always be called!)
        if cap["_statement_novelty"]:  # == previous claims!
            thoughts["no_statement_novelty"] = (
                "_statement_novelty",
                cap["_statement_novelty"],
            )
        else:
            thoughts["statement_novelty"] = (
                "_statement_novelty",
                cap["_statement_novelty"],
            )

    if "_overlaps" in filter:
        # Any single overlap?, e.g. 'overlap animal'
        if cap["_overlaps"]["_subject"]:
            for overlap in cap["_overlaps"]["_subject"]:
                overlap_name = "overlap -subj %s" % overlap["_entity"]["_types"][-1]
                thoughts[overlap_name] = (
                    "_overlaps",
                    {"_subject": [overlap], "_complement": []},
                )

        if cap["_overlaps"]["_complement"]:
            for overlap in cap["_overlaps"]["_complement"]:
                overlap_name = "overlap -compl %s" % overlap["_entity"]["_types"][-1]
                thoughts[overlap_name] = (
                    "_overlaps",
                    {"_subject": [], "_complement": [overlap]},
                )

        # Any pairs of overlaps?, e.g. 'overlap animal person'
        if cap["_overlaps"]["_subject"]:
            for overlaps in combinations(cap["_overlaps"]["_subject"], r=2):
                entities = sorted(
                    [overlaps[0]["_entity"]["_types"][-1], overlaps[1]["_entity"]["_types"][-1]]
                )
                overlap_name = "overlap -subj %s %s" % (entities[0], entities[1])
                thoughts[overlap_name] = (
                    "_overlaps",
                    {"_subject": overlaps, "_complement": []},
                )

        if cap["_overlaps"]["_complement"]:
            for overlaps in combinations(cap["_overlaps"]["_complement"], r=2):
                entities = sorted(
                    [overlaps[0]["_entity"]["_types"][-1], overlaps[1]["_entity"]["_types"][-1]]
                )
                overlap_name = "overlap -compl %s %s" % (entities[0], entities[1])
                thoughts[overlap_name] = (
                    "_overlaps",
                    {"_subject": [], "_complement": overlaps},
                )

    if "_entity_novelty" in filter:
        # Any entity novelties?
        if cap["_entity_novelty"]["_subject"] == "True":
            novelty_name = (
                    "entity_novelty -subj %s" % utt["triple"]["_subject"]["_types"][0]
            )
            novelty_info = {"_subject": True, "_complement": False}
            thoughts[novelty_name] = ("_entity_novelty", novelty_info)

        if cap["_entity_novelty"]["_complement"] == "True":
            novelty_name = (
                    "entity_novelty -compl %s" % utt["triple"]["_complement"]["_types"][0]
            )
            novelty_info = {"_subject": False, "_complement": True}
            thoughts[novelty_name] = ("_entity_novelty", novelty_info)

        # Alternative, if there are no novel entities
        novelty_info = {"_subject": False, "_complement": False}
        thoughts["entity_novelty -none"] = ("_entity_novelty", novelty_info)

    if "_subject_gaps" in filter:
        # Any subject gaps?, e.g. 'subject_gap person animal'
        if cap["_subject_gaps"]["_subject"]:
            for gap in cap["_subject_gaps"]["_subject"]:
                gap_name = "subject_gap -subj %s %s" % (
                    utt["triple"]["_subject"]["_types"][0],
                    gap["_entity"]["_types"][-1],
                )
                thoughts[gap_name] = ("_subject_gaps", {"_subject": [gap], "_complement": []})

        if cap["_subject_gaps"]["_complement"]:
            for gap in cap["_subject_gaps"]["_complement"]:
                gap_name = "subject_gap -compl %s %s" % (
                    utt["triple"]["_subject"]["_types"][0],
                    gap["_entity"]["_types"][-1],
                )
                thoughts[gap_name] = ("_subject_gaps", {"_subject": [], "_complement": [gap]})

        # Alternative, if there is no subject gap
        thoughts["subject_gap -none"] = (
            "_subject_gaps",
            {"_subject": [], "_complement": []},
        )

    if "_complement_gaps" in filter:
        # any object gaps?, e.g. 'object_gap person animal'
        if cap["_complement_gaps"]["_subject"]:
            for gap in cap["_complement_gaps"]["_subject"]:
                gap_name = "object_gap -subj %s %s" % (
                    utt["triple"]["_complement"]["_types"][0],
                    gap["_entity"]["_types"][-1],
                )
                thoughts[gap_name] = (
                    "_complement_gaps",
                    {"_subject": [gap], "_complement": []},
                )

        if cap["_complement_gaps"]["_complement"]:
            for gap in cap["_complement_gaps"]["_complement"]:
                gap_name = "object_gap -compl %s %s" % (
                    utt["triple"]["_complement"]["_types"][0],
                    gap["_entity"]["_types"][-1],
                )
                thoughts[gap_name] = (
                    "_complement_gaps",
                    {"_subject": [], "_complement": [gap]},
                )

        # Alternative, if there is no object gap
        thoughts["object_gap -none"] = (
            "_complement_gaps",
            {"_subject": [], "_complement": []},
        )

    if "_complement_conflict" in filter:
        # Any complement conflicts (cardinality conflict)?
        if cap["_complement_conflict"]:
            thoughts["complement_conflict"] = (
                "_complement_conflict",
                cap["_complement_conflict"][:1],
            )

    if "_negation_conflicts" in filter:
        # A negation conflict?
        if cap["_negation_conflicts"]:
            positives = [
                item
                for item in cap["_negation_conflicts"]
                if item["_polarity_value"] == "POSITIVE"
            ]
            negatives = [
                item
                for item in cap["_negation_conflicts"]
                if item["_polarity_value"] == "NEGATIVE"
            ]

            if positives and negatives:
                conflict_info = [random.choice(positives), random.choice(negatives)]
                thoughts["negation_conflict"] = ("_negation_conflicts", conflict_info)

    # Scramble to break ordering!
    thoughts = list(thoughts.items())
    random.shuffle(thoughts)
    return dict(thoughts)
