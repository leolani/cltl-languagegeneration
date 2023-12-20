import json

from tqdm import tqdm

from cltl.reply_generation.simplenlg_phraser import SimplenlgPhraser

# Read scenario from file
scenario_file_name = 'random-selections.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = SimplenlgPhraser()

for brain_response in tqdm(scenario):
    print(f"\n\n---------------------------------------------------------------\n")
    reply = replier.reply_to_statement(brain_response, persist=True)

    if not reply:
        reply = "NO REPLY GENERATED"

    print(f"Utterance by {brain_response['statement']['author']['label']}: {brain_response['statement']['utterance']}")
    print(f"Reply: {reply}")
