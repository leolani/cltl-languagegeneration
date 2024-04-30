import json


def get_triple_text_from_statement (statement):
    triple = statement["triple"]
    triple_text = triple["_subject"]["_label"]+", "+triple["_predicate"]["_label"]+", "+triple["_complement"]["_label"]
    return triple_text

def get_perspective_from_statement  (statement):
    perspective = statement["perspective"]
    perspective_text = ""
    if not perspective['_certainty']=='UNDERSPECIFIED':
        perspective_text += perspective['_certainty']+ ", "
    if perspective['_polarity']=='POSITIVE':
        perspective_text += 'believes'+ ", "
    elif perspective['_polarity']=='NEGATIVE':
        perspective_text += 'denies'+ ", "
    if not perspective['_sentiment']=='NEUTRAL':
        perspective_text += perspective['_sentiment']+ ", "
    if not perspective['_emotion']=='UNDERSPECIFIED':
        perspective_text += perspective['_emotion']+ ", "
    return perspective_text

def get_source_from_statement (statement):
    #"author": {"label": "piek"
    author = statement["author"]["label"]
    return author
