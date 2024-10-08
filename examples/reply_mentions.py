from cltl.reply_generation.lenka_replier import LenkaReplier
import json
from tqdm import tqdm

# Read scenario from file
scenario_file_name = 'basic-mentions-responses.json'
scenario_json_file = './data/' + scenario_file_name

f = open(scenario_json_file, )
scenario = json.load(f)

replier = LenkaReplier()

for brain_response in tqdm(scenario):
    reply = replier.reply_to_mention(brain_response, persist=True)
    print(reply)
