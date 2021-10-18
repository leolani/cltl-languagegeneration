from cltl.brain.long_term_memory import LongTermMemory
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat, UtteranceHypothesis

brain = LongTermMemory(clear_all=False)
chat = Chat("Lenka")
replier = LenkaReplier()

# one or several questions are queried in the brain
chat.add_utterance([UtteranceHypothesis("This is a test", 1.0)])
chat.last_utterance.analyze()
brain_response = brain.query_brain(chat.last_utterance, reason_types=True)
reply = replier.reply_to_question(brain_response)
print(reply)
