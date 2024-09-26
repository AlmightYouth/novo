from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Union, List, Optional, Tuple
from transformers import AutoTokenizer, Cache, DynamicCache

@dataclass
class OutputStruct:
    logits : Optional[Tensor] = None
    kv_cache : Optional[Union[Cache,DynamicCache]] = None
    hidden_states : Optional[Tensor] = None
    head_norms : Optional[Tensor] = None
    attn_map : Optional[Tensor] = None 
    value : Optional[Tensor] = None 
    loss: Optional[float] = None

class MixinDecoderCausalLM:
    def __init__(self, config) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

    def tokenise(self, s : str) -> Tensor:
        return self.tokenizer.encode(s,return_tensors='pt').to(self.device)
    
    @torch.no_grad()
    def infer_forward(
        self, 
        input_ids : Union[Tensor, str], 
        output_norms: bool = True, 
        **kwargs) -> OutputStruct:

        if isinstance(input_ids, str):
            input_ids = self.tokenise(input_ids)

        return self(input_ids,output_norms=output_norms,**kwargs)
    
    def zshot_classify(
            self, 
            prompt: str, 
            choices: List[str], 
            indices: List[Tensor],
            return_scores: bool = False
            ) -> Union[int,Tensor]:

        head_norms = []
        for c in choices:
            tokens = self.tokenise(prompt+" "+c)
            hn = self.infer_forward(tokens).head_norms[0,-1,:,:].detach().cpu()
            head_norms.append(hn)
        head_norms = torch.stack(head_norms).flatten(1)

        if return_scores:
            return head_norms

        individual_preds = torch.cat([
            head_norms[:,indices[0]].argmax(0),
            head_norms[:,indices[1]].argmin(0)])
        
        pred = torch.mode(individual_preds).values.item()

        return pred
