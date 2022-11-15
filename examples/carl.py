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
    reply = None
    if 'statement' in brain_response:
        reply = replier.reply_to_statement(brain_response, persist=True,
                                       thought_options=['_subject_gaps', '_complement_gaps'])
    elif 'question' in brain_response:
        reply = replier.reply_to_question(brain_response)
    elif 'mention' in brain_response:
        reply = replier.reply_to_mention(brain_response, persist=True)
    elif 'experience' in brain_response:
        reply = "No replies to EXPERIENCE have been implemented"

    print(reply)
