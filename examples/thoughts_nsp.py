import json

from tqdm import tqdm

from cltl.reply_generation.nsp_replier import NSPReplier

# Read scenario from file
scenario_file_name = 'thoughts-responses.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = NSPReplier('./../src/cltl/reply_generation/thought_selectors/nsp_model')

for brain_response in tqdm(scenario):

    print(f"\n\n---------------------------------------------------------------\n")
    reply = replier.reply_to_statement(brain_response, proactive=True, persist=True)

    if not reply:
        reply = "NO REPLY GENERATED"

    print(f"\nUtterance: {brain_response['statement']['utterance']}")
    print(f"Reply: {reply}")
