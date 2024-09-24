
from typing import Tuple,List, Dict
from os.path import join as osjoin
import sys
sys.path.append("..")
from normie.utils import pickle_rw

def get_dataset(name : str) -> Tuple[List[Dict], str]:

    mappings = {
        'sst2' : "Given a movie review sentence, determine if the sentiment is positive or negative.",
        'qqp' : "Are Questions 1 and 2 paraphrases of each other and semantically equivalent?",
        'mnli' : (
            "Natural Langauge Inference: Given a premise and a hypothesis, classify the relationship as entailment, contradiction, or neutral. "
            "Use your language understanding abilities to infer the relationship based on general knowledge and the context provided."),
        'mnli-mm' : (
            "Natural Langauge Inference: Given a premise and a hypothesis, classify the relationship as entailment, contradiction, or neutral. "
            "Use your language understanding abilities to infer the relationship based on general knowledge and the context provided."),
        'qnli': "Read and understand the Question and Context sentences. Determine if the context contains the answer to the question.",
        'rte': "Recognizing Textual Entailment: using your linguistic skills, nuanced understanding and real-world knowledge, determine if Sentence 2 is an entailment of Sentence 1.",
        'arce': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully. "),
        'tqa': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully. "
            "If you are unsure, you may respond with no comment."),
        'csqa2': (
            "Evaluate the question and apply commonsense reasoning "
            "to select the most plausible answer from the provided choices. "
            "Rely on implicit world knowledge and logical inference to "
            "determine the answer that best fits the context of the question. "
            "Do not add any preambles, introductions or explanations."),
        'qasc': (
            "Read both facts 1 and 2, together with the question."
            "Read the question and select the option that best represents the correct answer to the question. "
            "Your answer to the question should be based on facts from the real world. "
            "Do not add any preambles, introductions or explanations."),
        'swag': (
            "Read the context sentence and complete the context sentence. "
            "Your sentence completion should be plausible and based on common sense and logical reasoning. "
            "Some context sentences are intentionally vague, which require knowledge about the real world to complete. "),
        'hellaswag': (
            "Read the context sentence and complete the context sentence. "
            "Your sentence completion should be plausible and based on common sense and logical reasoning. "
            "Some context sentences are intentionally vague, which require knowledge about the real world to complete. "),
        'siqa': (
            "Answer the question by using common sense, knowledge of acceptable human social behaviour, and logical reasoning. "
            "Some questions are intentionally vague, which require knowledge about the real world to answer. "),
        'piqa': (
            "Answer the question truthfully with facts from the real world while avoiding being misled. "
            "Some questions are intentionally misleading, some require knowledge about numerical facts, "
            "others are common misconceptions. Watch out for these pitfalls, and answer truthfully."),
        'cosmosqa': (
            "Read the context and question. "
            "The context consists of everyday narratives. "
            "Answer the question by selecting the option that best reflects the likely causes or effects of events in the context. "
            "Do not add any preambles, introductions or explanations."),
        'cicero': (
            "You are presented with a question, target and context. "
            "The question will ask about the contents of the target, such as its consequences or causes. "
            "To answer the question correctly, read the dialogue given in the context (demarcated as utterances utt) between persons A and B. "
            "use the dialogue given in the context, together with conversational reasoning, logic, and facts from the real world to answer the question about the target correctly. "
            "Do not add any preambles, introductions or explanations."),
        'cicero2': (
            "You are presented with a question, target and context. "
            "The question will ask about the contents of the target, such as its consequences or causes. "
            "To answer the question correctly, read the dialogue given in the context (demarcated as utterances utt) between persons A and B. "
            "use the dialogue given in the context, together with conversational reasoning, logic, and facts from the real world to answer the question about the target correctly. "
            "Do not add any preambles, introductions or explanations."),
    }
    
    inst = mappings[name]
    ds = pickle_rw(osjoin('data','datasets.p'))[name]
    return ds, inst