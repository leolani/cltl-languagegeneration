from cltl.commons.triple_helpers import filtered_types_names


def replace_pronouns(speaker, author=None, entity_label=None, role=None):
    if entity_label is None and author is None:
        return speaker

    if role == 'pos':
        # print('pos', speaker, entity_label)
        if speaker.lower() == entity_label.lower():
            pronoun = 'your'
        elif entity_label.lower() == 'leolani':
            pronoun = 'my'
        else:
            pronoun = entity_label  # third person pos.
        return pronoun

    # Fix author
    elif author is not None and author.lower() not in ['', 'unknown', 'none']:
        if speaker.lower() == author.lower():
            pronoun = 'you'
        elif author.lower() == 'leolani':
            pronoun = 'I'
        else:
            pronoun = author.title()

        return pronoun

    # Entity
    if entity_label is not None and entity_label.lower() not in ['', 'unknown', 'none']:
        if speaker.lower() in [entity_label.lower(), 'speaker'] or entity_label == 'Speaker':
            pronoun = 'you'
        elif entity_label.lower() == 'leolani':
            pronoun = 'I'
        else:
            pronoun = entity_label

        return pronoun


def assign_spo(utterance, item):
    empty = ['', 'unknown', 'none']

    # INITIALIZATION

    if utterance['predicate']['label'] is None or utterance['predicate']['label'].lower() in empty:
        predicate = item['pOriginal']['value']
    else:
        predicate = utterance['predicate']['label']
    if utterance['subject']['label'] is None or utterance['subject']['label'].lower() in empty:
        subject = item['slabel']['value']
    else:
        subject = utterance['subject']['label']

    if utterance['object']['label'] is None or utterance['object']['label'].lower() in empty:
        object = item['olabel']['value']
    else:
        object = utterance['object']['label']

    return subject, predicate, object


def deal_with_authors(author, previous_author, predicate, previous_predicate, say):
    # Deal with author
    if not author:
        author = "someone"

    if author != previous_author:
        say += author + ' told me '
        previous_author = author
    else:
        if predicate != previous_predicate:
            say += ' that '

    return say, previous_author


def fix_entity(entity, speaker):
    new_ent = ''
    if entity and '-' in entity:
        entity_tokens = entity.split('-')

        for word in entity_tokens:
            new_ent += replace_pronouns(speaker, entity_label=word, role='pos') + ' '

    else:
        new_ent += replace_pronouns(speaker, entity_label=entity)

    entity = new_ent
    return entity


def clean_overlaps(overlaps):
    # Clean duplicates in overlaps
    overlapss = []
    seen = set()
    for ov in overlaps:
        if not ov['_entity']['_id'] in seen:
            seen.add(ov['_entity']['_id'])
            overlapss.append(ov)

    return overlapss


def any_type(utterance):
    if 'person' in filtered_types_names(utterance['triple']['_complement']['_types']):
        any_type = 'anybody'
    elif 'location' in filtered_types_names(utterance['triple']['_complement']['_types']):
        any_type = 'anywhere'
    else:
        any_type = 'anything'

    return any_type



def dash_replace(triple_text):
    return triple_text.replace(' ', '-')


def prepare_triple(utterance):
    return f"{dash_replace(utterance['triple']['_subject']['_label'])} " \
           f"{dash_replace(utterance['triple']['_predicate']['_label'])} " \
           f"{dash_replace(utterance['triple']['_complement']['_label'])}"


def prepare_perspective(utterance):
    perspective = utterance['perspective']
    perspective_text = ""

    if '_polarity' in perspective.keys() and perspective['_polarity'].upper() == 'POSITIVE':
        perspective_text += 'CONFIRM,'
    elif '_polarity' in perspective.keys() and perspective['_polarity'].upper() == 'NEGATIVE':
        perspective_text += 'DENY,'

    if '_certainty' in perspective.keys() and not perspective['_certainty'].upper() == 'UNDERSPECIFIED':
        perspective_text += perspective['_certainty'].upper() + ", "

    if '_sentiment' in perspective.keys() and perspective['_sentiment'].upper() not in ['NEUTRAL', 'UNDERSPECIFIED']:
        perspective_text += perspective['_sentiment'].upper() + " SENTIMENT, "

    if '_emotion' in perspective.keys() and not perspective['_emotion'].upper() == 'UNDERSPECIFIED':
        perspective_text += perspective['_emotion'].upper() + " , "

    return perspective_text

def prepare_speaker(utterance):
    return f"{dash_replace(utterance['author']['label'])}"


def prepare_author_from_thought(thought):
    return f"{dash_replace(thought['_provenance']['_author']['_label'])}"


def prepare_gap(gap, role="_subject"):
    if role == "_subject":
        return f"{dash_replace(gap['_known_entity']['_label'])} " \
               f"{dash_replace(gap['_predicate']['_label'])} " \
               f"{dash_replace(filtered_types_names(gap['_target_entity_type']['_types']).upper())}"
    elif role == "_complement":
        return f"{dash_replace(filtered_types_names(gap['_target_entity_type']['_types']).upper())} " \
               f"{dash_replace(gap['_predicate']['_label'])} " \
               f"{dash_replace(gap['_known_entity']['_label'])}"


def prepare_overlap(utterance, overlap, role="_subject"):
    if role == "_subject":
        return f"{dash_replace(utterance['triple']['_subject']['_label'])} " \
               f"{dash_replace(utterance['triple']['_predicate']['_label'])} " \
               f"{dash_replace(overlap['_entity']['_label'])}"
    elif role == "_complement":
        return f"{dash_replace(overlap['_entity']['_label'])} " \
               f"{dash_replace(utterance['triple']['_predicate']['_label'])} " \
               f"{dash_replace(utterance['triple']['_complement']['_label'])}"