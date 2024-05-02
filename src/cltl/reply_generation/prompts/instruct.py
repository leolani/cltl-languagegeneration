import cltl.reply_generation.prompts.response_processing as processor

# _statement_novelty
# _entity_novelty
# _negation_conflicts
# _complement_conflict
# _subject_gaps
# _complement_gaps
# _overlaps
# _trust

LANGUAGE = "English"

instruct_for_statement = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a phrase, followed by a perspective, followed by \"that\" and a triple with a subject, predicate and object.\
     You need to paraphrase the input in plain " + LANGUAGE + ". \
     Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
}


instruct_for_subject_gap = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a triple with a subject, a predicate and a type of subject.\
     You need to paraphrase the input in plain " + LANGUAGE + " as a question for the object. \
     Use who for the type person, where for the type location, when for the type time and what for everything else. \
     Only reply with the short paraphrase of the input. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
           }

instruct_for_object_gap = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a triple with a subject, a predicate and a type of object.\
     You need to paraphrase the input in plain  " + LANGUAGE + "  as a question for the object. \
     Use who for the type person, where for the type location, when for the type time and what for everything else. \
     Only reply with the short paraphrase of the input. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
           }

instruct_for_novelty = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a triple with a subject, a predicate and a type of object and a phrase.\
     You need to paraphrase the input in plain  " + LANGUAGE + ". \
     Only reply with the short paraphrase of the input. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
           }

instruct_for_conflict = {"role": "system", "content": "You are an intelligent assistant. \
     I will give you as input: a triple with a subject, a predicate and a type of object and a phrase that expresses a conflict.\
     You need to paraphrase the input in plain  " + LANGUAGE + ". \
     Only reply with the short paraphrase of the input. \
     Do not give an explanation. \
     Do not explain what the subject and object is. \
     The response should be just the paraphrased text and nothing else."
           }

def create_statement_prompt(instruct:None, statement: str):
    triple_text = processor.get_triple_text_from_statement(statement)
    print("\n\nTriple text:", triple_text)
    perspective_text = processor.get_perspective_from_statement(statement)
    print('perspective_text', perspective_text)
    author = processor.get_source_from_statement(statement)
    print('author', author)
    prompt = [instruct, {"role": "user", "content": author + " is " + perspective_text + " that " + triple_text}]
    return prompt

