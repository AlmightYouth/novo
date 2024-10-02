import os.path as osp
from os.path import join as osjoin
from typing import Union, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from models.custom_deberta import DebertaV2Model
from utils import pickle_rw, strip_add_fullstop
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import softmax

def get_dataloaders(
        dataset_name: str, bsz: int, tokenizer: PreTrainedTokenizerBase
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """returns train, dev, test loaders that outputs tok,att,seg,lbl
    all of shape [bsz, nchoices, len_sample with padding], except lbl which
    is shaped [bsz,]
    """

    pad_token_id = tokenizer.pad_token_id
    sep = tokenizer.sep_token
    assert isinstance(pad_token_id,int) and isinstance(sep,str)

    results = {}

    # first tokeniser entire dataset

    ds = pickle_rw(f'datav2/{dataset_name}.p')
    for split in ['train','dev','test']:
        labels = []
        buffer = []
        for d in ds[split]:
            qns = d['question']
            for c in d['choices']:
                c = strip_add_fullstop(c)
                if dataset_name == 'swag':
                    buffer.append(f"{qns} {c}")
                else:
                    buffer.append(f"{qns} {sep} {c}")
            labels.append(d['label'])
        
        ed = tokenizer.batch_encode_plus(
            buffer,return_tensors='pt',padding=True,truncation=True)
        labels = torch.tensor(labels,dtype=torch.int64)

        # bsz, len(choices), len
        nchoices = len(d['choices'])
        len_sample = ed['input_ids'].size(-1)
        new_shape = (-1,nchoices,len_sample)
            
        tokens = ed['input_ids'].reshape(new_shape) 
        types = ed['token_type_ids'].reshape(new_shape) 
        attmask = ed['attention_mask'].reshape(new_shape)

        results[split] = DataLoader(
            TensorDataset(tokens,attmask,types,labels),
            bsz, shuffle=(split=='train'),num_workers=8,pin_memory=True)
    
    return results['train'], results['dev'], results['test']

def preprocess_flatten_input(tok, att, seg) -> Tuple[Tensor,Tensor,Tensor,int,int]:
    """for model forward.
    Flatten from (bsz, nchoices, len sample) to (bsz*nchoices, lensample).
    will revert shape back to original during classification
    """
    bsz, nchoices, _ = tok.shape
    tok = tok.flatten(0,1)
    att = att.flatten(0,1)
    seg = seg.flatten(0,1)
    return tok, att, seg, bsz, nchoices

def resolve_write_testpred(model: str, dataset: str, prob: Tensor, labels: Tensor):
    preds_pt = torch.argmax(prob,-1)
    preds = preds_pt.tolist()

    # yes/no per line. extension unspecified, maybe lst or txt
    if dataset == 'csqa2': 
        towrite = ['yes' if x == 0 else 'no' for x in preds]
        ext = 'lst'

    # headerless csv file with 2 cols. qid, pred (A,B,..)
    elif dataset == 'qasc':
        ids = [x['id'] for x in pickle_rw('datav2/qasc.p')['test']]
        towrite = [f"{_i},{chr(65+x)}" for (_i,x) in zip(ids,preds)]
        ext = 'csv'

    # single column csv file with 'pred' header, following by integer 0-3
    elif dataset == 'swag':
        towrite = ['pred']+preds
        ext = 'csv'
    
    # csv file with "annot_id,ending0,ending1,ending2,ending3" headers,
    # annot-id is test-0, test-1. endings0-3 are probability scores
    elif dataset == 'hellaswag':
        towrite = ['annot_id,ending0,ending1,ending2,ending3']
        for i,p in enumerate(prob.tolist()):
            p = [str(x) for x in p]
            towrite.append(','.join([f'test-{i}',*p]))
        ext = 'csv'
    
    # extension unspecified
    elif dataset == 'siqa':
        towrite = preds
        ext = 'lst'

    # extension unspecified
    elif dataset == 'piqa':
        towrite = preds
        ext = 'lst'
    
    # lst file with one pred per line
    elif dataset == 'cosmosqa':
        towrite = preds
        ext = 'lst'
    
    # meaningful labels
    elif 'cicero' in dataset: 
        ext = 'txt'
        acc = ((preds_pt==labels).sum(0)/len(preds_pt)).item()
        acc = round(acc*100,2)
        towrite = [acc]

    else: raise

    fpath = f"{model}-{dataset}-test-predictions.{ext}"
    towrite = [f"{x}\n"for x in towrite]
    with open(fpath,'w') as f:
        f.writelines(towrite)
    print(f"test predictions written to {fpath}!")
    return

def run_test_deberta(dataset_names: Union[str,List[str]], gpu_id: int = 1) -> float:
    """predictions are written to a filepath in a dataset-specific format."""
    
    if isinstance(dataset_names,str):
        dataset_names = [dataset_names]
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    for dname in dataset_names:
        
        _,_,testgen = get_dataloaders(dname,32,tokenizer,None)

        # load model
        root = osp.join('models',f'deberta-{dname}')
        plm = DebertaV2Model.from_pretrained(osjoin(root,'hf_plm'),device_map=f'cuda:{gpu_id}').eval()
        classifier = torch.load(osjoin(root,'classifier.pt')).cuda(gpu_id).eval()

        # inference loop
        all_labels = []
        all_logits = []
        for tok, att, seg, lbl in tqdm(testgen,desc=f'test {dname}'):
            tok, att, seg, bsz, nchoices = preprocess_flatten_input(tok,att,seg)
            with torch.no_grad():
                # head shaped (bsz*nchoices, seq, nlay, nhead)
                _, head_norms = plm(tok,att,seg,output_norms=True) 
                _, _, nlay, nhead = head_norms.shape

                # classifier expects (bsz, nchoices, seq, nlay*nhead)
                head_norms = head_norms.reshape(bsz, nchoices,-1, nlay*nhead)
                logits = classifier(head_norms)

            all_labels.append(lbl)
            all_logits.append(logits)
        
        # collate
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits).clone().detach().cpu()

        probabilities = softmax(all_logits,-1)

        resolve_write_testpred('deberta',dname,probabilities,all_labels)