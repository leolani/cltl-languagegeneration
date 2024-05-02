import cltl.reply_generation.prompts.instruct as instruct

# * _statement_novelty
# * _entity_novelty
# * _negation_conflicts
# * _complement_conflict
# * _subject_gaps
# * _complement_gaps
# * _overlaps
# * _trust




def get_prompt_input_from_response(response):
    statement = response["statement"]
    statement_text = get_triple_text_from_statement(statement)
    statement_author = get_author_from_statement(statement)
    prompts = []
    thought = response["thoughts"]

    novelties = thought["_statement_novelty"]
    for novelty in novelties:
        input = statement_author+" claims "+ statement_text+". Also "+get_provenance_from_statement_novelty(novelty["_provenance"])
        prompt = [instruct.instruct_for_novelty, {"role": "user", "content": input}]
        prompts.append(prompt)

    # @TODO
    # novelties = thought["_entity_novelty"]
    # for novelty in novelties:
    #     input = statement_author+" claims "+ statement_text+". Also "+get_XXX_from_entity_novelty(novelty)
    #     prompt = [instruct.instruct_for_novelty, {"role": "user", "content": input}]
    #     prompts.append(prompt)

    conflicts = thought["_negation_conflicts"]
    for conflict in conflicts:
        input = statement_author+" claims "+ statement_text+". But "+get_negation_conflict(conflict)
        prompt = [instruct.instruct_for_novelty, {"role": "user", "content": input}]
        prompts.append(prompt)

    # @TODO
    # conflicts = thought["_complement_conflict"]
    # for conflict in conflicts:
    #     input = statement_author+" claims "+ statement_text+". Also "+get_complement_conflict(conflict)
    #     inputs.append(input)

    gaps = thought["_subject_gaps"]
    if gaps["_subject"]:
        for gap in gaps["_subject"]:
            input = get_subject_gap_subject(gap)
            prompt = [instruct.instruct_for_subject_gap, {"role": "user", "content": input}]
            prompts.append(prompt)

        if gaps["_complement"]:
            for gap in gaps["_complement"]:
                input = get_subject_gap_complement(gap)
                prompt = [instruct.instruct_for_subject_gap, {"role": "user", "content": input}]
                prompts.append(prompt)

    gaps = thought["_complement_gaps"]
    if gaps["_subject"]:
        for gap in gaps["_subject"]:
            input = get_complement_gap_subject(gap)
            prompt = [instruct.instruct_for_subject_gap, {"role": "user", "content": input}]
            prompts.append(prompt)

        if gaps["_complement"]:
            for gap in gaps["_complement"]:
                input = get_complement_gap_complement(gap)
                prompt = [instruct.instruct_for_subject_gap, {"role": "user", "content": input}]
                prompts.append(prompt)

    # @TODO
    # overlaps = thought["_overlaps"]
    # for overlap in overlaps:
    #     input = statement_author+" claims "+ statement_text+". Also "+get_overlap_statement(overlap)
    #     inputs.append(input)

    # @TODO
    # trust = thought["_trust"]
    # input = statement_author+" claims "+ statement_text+". Also "+get_trust_statement(trust)
    # inputs.append(input)
    return prompts

def get_utterance_from_statement (statement):
    utterance = statement["utterance"]
    return utterance

def get_triple_text_from_statement (statement):
    triple = statement["triple"]
    triple_text = triple["_subject"]["_label"]+", "+triple["_predicate"]["_label"]+", "+triple["_complement"]["_label"]
    return triple_text

def get_perspective_from_statement  (statement):
    perspective = statement["perspective"]
    perspective_text = ""
    if not perspective['_certainty']=='UNDERSPECIFIED':
        perspective_text += perspective['_certainty'].tolower()+ ", "
    if perspective['_polarity']=='POSITIVE':
        perspective_text += 'believes'+ ", "
    elif perspective['_polarity']=='NEGATIVE':
        perspective_text += 'denies'+ ", "
    if not perspective['_sentiment']=='NEUTRAL':
        perspective_text += perspective['_sentiment']+ ", "
    if not perspective['_emotion']=='UNDERSPECIFIED':
        perspective_text += perspective['_emotion'].tolower()+ ", "
    return perspective_text

def get_author_from_statement (statement):
    author = statement["author"]["label"]
    return author

def get_subject_gap_subject(thought):
    known_entity = thought["_known_entity"]["_label"]
    predicate = thought["_predicate"]["_label"]
    gap_type = thought["_entity"]["_types"]
    gap_text = known_entity +", "+predicate+", "+ gap_type[0]
    return gap_text

def get_subject_gap_complement(thought):
    known_entity = thought["_known_entity"]["_label"]
    predicate = thought["_predicate"]["_label"]
    gap_type = thought["_entity"]["_types"]
    gap_text = gap_type[0]+", "+predicate+", "+known_entity
    return gap_text

def get_complement_gap_subject(thought):
    known_entity = thought["_known_entity"]["_label"]
    predicate = thought["_predicate"]["_label"]
    gap_type = thought["_entity"]["_types"]
    gap_text =gap_type[0]+", ", predicate+", "+known_entity
    return gap_text

def get_complement_gap_complement(thought):
    known_entity = thought["_known_entity"]["_label"]
    predicate = thought["_predicate"]["_label"]
    gap_type = thought["_entity"]["_types"]
    gap_text = known_entity +", " + predicate+ ", " + gap_type[0]
    return gap_text

def get_negation_conflict(conflict):
    provenance = conflict["_provenance"]
    author = provenance["_author"]["_label"]
    date = provenance["_date"]
    value = conflict["_polarity_value"]
    if value == "NEGATIVE":
        value = "denies"
    elif value == "POSITIVE":
        value = "claims"
    novelty_text = author + " "+ value+ " me on " + date
    return novelty_text


# def get_cardinality_conflict(thought):
#     author = thought["_author"]["label"]
#     date = thought["_date"]
#     value = thought["_polarity_value"]
#     if value == "NEGATIVE":
#         value = "denies"
#     elif value == "POSITIVE":
#         value = "claims"
#     novelty_text = author + " "+ value+ " me on " + date
#     return novelty_text

def get_provenance_from_statement_novelty(provenance):
    # {'_provenance': {
    #     '_author': {'_id': 'http://cltl.nl/leolani/friends/carl', '_label': 'carl', '_offset': None, '_confidence': 0.0,
    #                 # '_types': ['Source', 'Actor']}, '_date': '2017-10-24'}}
    author = provenance["_author"]["_label"]
    date = provenance["_date"]
    novelty_text = author + " told me this on "+ date
    return novelty_text


