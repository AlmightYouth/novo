import torch
import numpy as np
from torch import Tensor
from typing import List, Dict, Optional
from models import get_model
from tqdm import tqdm
from utils import strip_add_fullstop, get_prompt, pickle_rw
import argparse

def finalise_head_indices(acc: Tensor, q: float) -> List[Tensor]:
    results = []

    thres = torch.quantile(acc, q, dim = -1)
    for i in range(2):
        results.append(torch.where(acc[i] > thres[i].item())[0])

    a0, a1 = [x.numpy() for x in results]
    dups = np.intersect1d(a0,a1)
    a0 = a0[~np.isin(a0, dups)]
    a1 = a1[~np.isin(a1, dups)]

    results = [torch.from_numpy(a0), torch.from_numpy(a1)]
    return results

def discovery(
        model_name: str, samples: List[Dict], 
        quantile_threshold: float = 0.85,
        inst: Optional[str] = None,
        gpu_id: int = 0
        ) -> List[Tensor]:
    """returns indices of useful heads."""

    model, pfmt = get_model(model_name,gpu_id)
    
    acc_arr = torch.zeros((2,model.config.num_hidden_layers*model.config.num_attention_heads))
    for d in tqdm(samples):
        prompt = get_prompt(pfmt, d['question'], inst)
        choices = strip_add_fullstop(d['choices'])
        head_norms = model.zshot_classify(prompt,choices,None,True)

        acc_arr[0] += (head_norms.argmax(0) == d['label']).int()
        acc_arr[1] += (head_norms.argmin(0) == d['label']).int()
    acc_arr = (acc_arr/len(samples))*100
    
    heads = finalise_head_indices(acc_arr, quantile_threshold)
    return heads


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['mistral-7b-it', 'llama2-7b-chat', 'llama2-7b', 'vicuna-7b'])
    parser.add_argument('--dataset', '-d', choices=['tqa','csqa2','qasc','swag','hellaswag','siqa','piqa','cosmosqa','cicero','cicero2'])
    parser.add_argument('--quantile_thres', default=0.85, type=float, help='[Optional] Specify the quantile threshold (defaults to 0.85)')   
    parser.add_argument('--gpu', default=0, type=int, help='[Optional] Specify which GPU to use for model inference.')
    args = parser.parse_args()

    samples = pickle_rw('heads.p')[args.model][args.dataset]['discovery_samples']
    heads = discovery(args.model, samples, args.quantile_thres, args.gpu)
    print(f"DISCOVERY COMPLETE FOR {args.model} | {args.dataset}")
    print('Indices of normal heads:')
    print(heads[0])
    print('Indices of inverted heads:')
    print(heads[1])
