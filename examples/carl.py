from cltl.reply_generation.lenka_replier import LenkaReplier
import json
from tqdm import tqdm

# Read scenario from file
scenario_file_name = 'carl-responses.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = LenkaReplier()

for brain_response in tqdm(scenario):
    reply = replier.reply_to_statement(brain_response, proactive=True, persist=True)
    print(reply)
