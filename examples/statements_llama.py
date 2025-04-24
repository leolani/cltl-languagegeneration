import json

from tqdm import tqdm

from cltl.reply_generation.llama_phraser import LlamaPhraser

def print_thought(thought):
    if type(thought) == dict:
        # entity novelty, gaps or overlaps
        pass
    elif type(thought) == list:
        # conflicts or statement novelty
        pass

# Read scenario from file
scenario_file_name = 'pertype-selections.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = LlamaPhraser()

for brain_response in tqdm(scenario):
    thought_type = list(brain_response['thoughts'].keys())[0]
    if brain_response['thoughts'][thought_type]:
        reply = replier.reply_to_statement(brain_response, persist=True, casefold=False)
        if not reply:
            reply = "NO REPLY GENERATED"
            continue
    else:
        reply = "NO THOUGHT"
        continue

    print(f"\n\n---------------------------------------------------------------\n")
    print(f"Utterance: {brain_response['statement']['utterance']}")
    print(f"Thought type: {list(brain_response['thoughts'].keys())[0]}")
    print(f"Reply: {reply}")
