import cltl.reply_generation.prompts.response_processing as processor

# _statement_novelty
# _entity_novelty
# _negation_conflicts
# _complement_conflict
# _subject_gaps
# _complement_gaps
# _overlaps
# _trust

instruct_for_statement = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a phrase, followed by a perspective, followed by \"that\" and a triple with a subject, predicate and object.\
     You need to paraphrase the input in plain English. \
     Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
}

def create_prompt(instruct:None, statement: str):
    triple_text = processor.get_triple_text_from_statement(statement)
    print("\n\nTriple text:", triple_text)
    perspective_text = processor.get_perspective_from_statement(statement)
    print('perspective_text', perspective_text)
    author = processor.get_source_from_statement(statement)
    print('author', author)
    prompt = [instruct, {"role": "user", "content": author + " is " + perspective_text + " that " + triple_text}]
    return prompt
