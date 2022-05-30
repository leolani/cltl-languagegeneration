import json

from tqdm import tqdm

from cltl.reply_generation.lenka_replier import LenkaReplier

# Read scenario from file
scenario_file_name = 'basic-questions-responses.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = LenkaReplier()

for brain_response in tqdm(scenario):

    print(f"\n\n---------------------------------------------------------------\n")
    reply = replier.reply_to_question(brain_response)

    if not reply:
        reply = "NO REPLY GENERATED"

    print(f"Utterance: {brain_response['question']['utterance']}")
    print(f"Reply: {reply}")
