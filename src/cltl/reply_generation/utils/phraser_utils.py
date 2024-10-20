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
