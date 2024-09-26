from typing import List, Optional
from models import get_model
from data import get_dataset
from utils import strip_add_fullstop, get_prompt, pickle_rw, get_heads, aliases
from tqdm import tqdm
from torch import Tensor
import argparse

def inference(
        model_name: str, 
        dataset_name: str, 
        heads: Optional[List[Tensor]] = None,
        gpu_id: int = 0) -> float:
    """Calculate zero-shot validation accuracies of the dataset of a given model, using head norms.    
    Arg:  
        `model_name`: mistral-7b-it, llama2-7b-chat, llama2-7b or vicuna-7b  
        `dataset_name`: tqa, csqa2, qasc, swag, hellaswag, siqa, piqa, cosmosqa, cicero, or cicero2  
        `heads`: A list containing two 1D int64 tensor. Can be left as None to load default heads. 
        `gpu_id`: which GPU to use.
    """
    dataset, inst = get_dataset(dataset_name)
    model, pfmt = get_model(model_name,gpu_id)

    acc = 0
    for d in tqdm(dataset):
        prompt = get_prompt(pfmt, d['question'], inst)
        choices = strip_add_fullstop(d['choices'])
        pred = model.zshot_classify(prompt,choices,heads)
        acc += int(pred == d['label'])
    acc /= len(dataset)
    print(f"{aliases[model_name]} | {aliases[dataset_name]} | Accuracy {acc:.2%}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['mistral-7b-it', 'llama2-7b-chat', 'llama2-7b', 'vicuna-7b'])
    parser.add_argument('--dataset', '-d', choices=['tqa','csqa2','qasc','swag','hellaswag','siqa','piqa','cosmosqa','cicero','cicero2'])
    parser.add_argument('--gpu', default=0, type=int, help='[Optional] Specify which GPU to use for model inference.')
    parser.add_argument('--heads', default=None, help='[Optional] Path to your custom head indices. Loaded with pickle.')
    args = parser.parse_args()

    if args.heads == None:
        args.heads = get_heads(args.model,args.dataset)
    else:
        args.heads = pickle_rw(args.heads)

    inference(args.model,args.dataset,args.heads,args.gpu)
    
