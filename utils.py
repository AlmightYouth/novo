from typing import List, Any,Callable, Optional
import pickle
from torch import Tensor

aliases = {
    'mistral-7b-it' : 'Mistral-7B-Instruct-v0.2',
    'llama2-7b' : 'Llama2-7B',
    'llama2-7b-chat': 'Llama2-7B-Chat',
    'vicuna-7b': 'Vicuna-7B-v1.5',
    'tqa' : 'TruthfulQA',
    'csqa2' : 'CommonSenseQA-2.0',
    'qasc' : 'QASC',
    'swag' : 'SWAG',
    'hellaswag' : 'HellaSwag',
    'siqa' : 'Social-IQA',
    'piqa' : 'Physical-IQA',
    'cosmosqa' : 'CosmosQA',
    'cicero' : 'CICERO v1',
    'cicero2' : 'CICERO v2'}

def get_heads(m: str, d: str) -> List[Tensor]:
    """loads and return the head indices for a model `m` and dataset `d`"""
    return pickle_rw('heads.p')[m][d]['heads']

def pickle_rw(path : str, mode : str = 'r', obj : Any = None) -> Any:
    if mode not in 'rw': raise
    if mode == 'w' and obj is None: raise
    if mode == 'r' and obj is not None: raise
    with open(path, f"{mode}b") as f:
        if mode == 'r':
            return pickle.load(f)
        else:
            pickle.dump(obj, f)

def strip_add_fullstop(choices : List[str]) -> List[str]:
    res = []
    for c in choices:
        c = c.strip()

        if not c.endswith('.'):
            c = c+"."

        res.append(c)
    return res

def get_prompt(format_prompt: Callable, qns: str, inst: Optional[str] = None) -> str:
    if inst is None:
        inst = ""
    return format_prompt(inst,qns)