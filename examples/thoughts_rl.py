import json

from tqdm import tqdm

from cltl.reply_generation.rl_replier import RLReplier

# Read scenario from file
scenario_file_name = 'thoughts-responses.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = RLReplier(None, './../src/cltl/reply_generation/reinforcement_learning/thoughts.json')

for brain_response in tqdm(scenario):
    replier.reward_thought()

    print(f"\n\n---------------------------------------------------------------\n")
    reply = replier.reply_to_statement(brain_response, proactive=True, persist=True)

    if not reply:
        reply = "NO REPLY GENERATED"

    print(f"\nUtterance: {brain_response['statement']['utterance']}")
    print(f"Reply: {reply}")
