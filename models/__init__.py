from .custom_mistral import MistralForCausalLM
from .custom_llama import LlamaForCausalLM
from .custom_deberta import DebertaV2ForSequenceClassification
from typing import Tuple, Callable, Optional, Union

def get_model(
        name : str, dvc : Optional[Union[str,int]] = None
        ) -> Tuple[MistralForCausalLM, Callable]:
    """returns model, prompt formatter"""
    
    if isinstance(dvc,int) and dvc >= 0:
        dvc = f"cuda:{dvc}"

    kwargs = {'torch_dtype':'auto','device_map':dvc}

    if name == 'mistral-7b-it':
        model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2',**kwargs)
        format_prompt = lambda s, m : f"[INST] {s} {m} [/INST]"

    elif name == 'vicuna-7b':
        model = LlamaForCausalLM.from_pretrained('lmsys/vicuna-7b-v1.5',**kwargs)
        format_prompt=lambda s, m : f"A chat between a user and an assistant. USER: {s} {m} ASSISTANT:"

    elif name == 'llama2-7b':
        model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',**kwargs)
        format_prompt=lambda s, m : f"{s}\n{m}"
    
    elif name == 'llama2-7b-chat':
        model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf',**kwargs)
        format_prompt=lambda s, m : f"[INST] <<SYS>>\n{s}\n<</SYS>>\n\n{m} [/INST]"
        
    else:
        raise ValueError(f"No such model {name}")

    return model, format_prompt

__all__ = [
    'MistralForCausalLM',
    'LlamaForCausalLM',
    'DebertaV2ForSequenceClassification', 
    'get_model',
    'OutputStruct'
    ]