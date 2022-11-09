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
