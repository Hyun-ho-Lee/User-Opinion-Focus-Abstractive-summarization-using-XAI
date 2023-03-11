import datasets
import torch
import pandas as pd 
import itertools 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from transformers import BartTokenizer, BartForConditionalGeneration,BartForSequenceClassification
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm
from torch.utils.data import DataLoader

from bartdataset_normal_min import get_loader

rouge = datasets.load_metric("rouge")
bscore = datasets.load_metric("bertscore")

_,_,test_loader = get_loader(128,0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_cf = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=2).to(device)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
new_ckpt = dict((k[6:], v) for (k, v) in torch.load('./final_baseline/10_2.7383.pt')['state_dict'].items())
model.load_state_dict(new_ckpt,strict=False)


generate_pred = [] 
gold_pred = []


for batch in tqdm(test_loader):
    doc = batch[0]
    target_ids = batch[2].to(device)
    summary_ids = model.generate(doc.cuda(),length_penalty=1.0,max_length= 15,early_stopping=True, num_beams=6, no_repeat_ngram_size=1)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    target = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
    rouge.add_batch(predictions=summary, references=target)
    bscore.add_batch(predictions=summary, references=target)
    with torch.no_grad():
        generate_logits = model_cf(summary_ids).logits
        generate_pred.append(generate_logits.argmax(axis=1).detach().cpu().tolist()) 
        


score = rouge.compute(rouge_types=['rouge1','rouge2','rougeL'], use_stemmer=True)
results = bscore.compute(model_type="distilbert-base-uncased", device='cuda')

print(((score['rouge1'].mid.fmeasure)*100,
      '\n',(score['rouge2'].mid.fmeasure)*100,
      '\n',(score['rougeL'].mid.fmeasure)*100))

print((np.mean(results['f1']))*100)



generate_pred_t = list(itertools.chain(*generate_pred))
gold_pred_t = list(itertools.chain(*gold_pred))
test = pd.read_csv('./test.csv')
test_true = test['overall']
test_true=test_true.tolist()

print("Generate Summary ACC:",accuracy_score(test_true, generate_pred_t)*100)
print("F1 Score Summary : ",f1_score(test_true, generate_pred_t, pos_label=1)*100)

