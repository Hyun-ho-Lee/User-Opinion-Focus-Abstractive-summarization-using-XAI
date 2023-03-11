#%%
import torch 
import numpy as np
import pandas as pd  
import random
import os 
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


from transformers import AutoTokenizer,AddedToken
from transformers import BartTokenizer

#%%
def seed_everything(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#%%


def add_to_end_token(x): 
    
    x= x + '</s>'
    
    return x



def load_data(shap_type):
    
    train= pd.read_csv('./train.csv')
    val = pd.read_csv('./val.csv')
    test = pd.read_csv('./test.csv')
    
    
    train['gold']  = train['summary'].apply(add_to_end_token)
    val['gold'] = val['summary'].apply(add_to_end_token)
    test['gold'] = test['summary'].apply(add_to_end_token)
    train= train[['review','summary','gold','overall']]
    val= val[['review','summary','gold','overall']]  
    test= test[['review','summary','gold','overall']]   
    
    with open(f"./shap_{shap_type}.pkl","rb") as fr:
        shap_data= pickle.load(fr)
        train['shap'] = shap_data
        train.dropna(inplace=True)
        train.isnull().sum()

    

    return train, val,test

train,val,test = load_data("median")


#%%

class CustomDataset(Dataset):
    def __init__(self, dataset, option):
        self.dataset = dataset 
        self.option = option
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        
        if self.option == 'val':
        
            row = self.dataset.iloc[idx, 0:4].values


            encoder_text = row[0]
            decoder_text = row[1]
            labels = row[2]
            overall = row[3]

    
            
            encoder_inputs = self.tokenizer(
            encoder_text,
            return_tensors='pt',
            max_length=512,
            pad_to_max_length=True,
            add_special_tokens=True,
            truncation = True)
        
            decoder_inputs = self.tokenizer(
            decoder_text,
            return_tensors='pt',
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=True,
            truncation = True)
        
            labels = self.tokenizer(
            labels,
            return_tensors='pt',
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=False,
            truncation = True)
        
            encoder_input_ids = encoder_inputs['input_ids'][0]
            encoder_attention_mask = encoder_inputs['attention_mask'][0]
            decoder_input_ids = decoder_inputs['input_ids'][0]
            decoder_attention_mask = decoder_inputs['attention_mask'][0]
            
            label = torch.cat([labels['input_ids'][0]], dim=0).unsqueeze(0)
            # We have to make sure that the PAD token is ignored for calculating the loss
            label = torch.IntTensor([[-100 if token == 0 else token.item() for token in label[0]]])
            labels = label.type(torch.LongTensor)
            labels = labels.squeeze(1)
            
            return encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,overall,labels
        else : 
            
            row = self.dataset.iloc[idx, 0:5].values


            encoder_text = row[0]
            decoder_text = row[1]
            labels = row[2]
            overall = row[3]
            shap = row[4]
            
            
            
            
            encoder_inputs = self.tokenizer(
            encoder_text,
            return_tensors='pt',
            max_length=512,
            pad_to_max_length=True,
            add_special_tokens=True,
            truncation = True)
        
            decoder_inputs = self.tokenizer(
            decoder_text,
            return_tensors='pt',
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=True,
            truncation = True)
        
            labels = self.tokenizer(
            labels,
            return_tensors='pt',
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=False,
            truncation = True)
            
            shap_mask= self.tokenizer(
            shap,
            return_tensors='pt',
            max_length=512,
            pad_to_max_length=True,
            add_special_tokens=True,
            truncation = True)
        
            encoder_input_ids = encoder_inputs['input_ids'][0]
            encoder_attention_mask = encoder_inputs['attention_mask'][0]
            decoder_input_ids = decoder_inputs['input_ids'][0]
            decoder_attention_mask = decoder_inputs['attention_mask'][0]
            shap_mask = shap_mask['input_ids'][0]

            shap_mask = torch.IntTensor([[0 if token == 1  else token.item()  for token in shap_mask]])
            shap_mask = shap_mask.type(torch.LongTensor)
            
            for idx in range(len(shap_mask[0])):
    
                if shap_mask[0][idx] != 0: 
                    
                    shap_mask[0][idx] = 1
            
                else :
                    
                    shap_mask[0][idx] = 0
                    
                shap_mask[0][0] = 1
                        
            

            label = torch.cat([labels['input_ids'][0]], dim=0).unsqueeze(0)
            # We have to make sure that the PAD token is ignored for calculating the loss
            label = torch.IntTensor([[-100 if token == 0 else token.item() for token in label[0]]])
            labels = label.type(torch.LongTensor)
            labels = labels.squeeze(1)
            
            

            
            
            return encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,shap_mask,overall,labels

    


def get_loader(batch_size,num_workers):

    
    train_loader = DataLoader(dataset=CustomDataset(train, 'train'),
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=CustomDataset(val, 'val'),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    test_loader = DataLoader(dataset=CustomDataset(test,'val'),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)
    
    return train_loader, val_loader , test_loader 


def get_dist_loader(batch_size,num_workers):

    
    train_dataset = CustomDataset(train, 'train')
    val_dataset = CustomDataset(val, 'val')
    #test_dataset = CustomDataset(test, 'test')
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    #test_sampler = DistributedSampler(test_dataset)
    
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=train_sampler,
                              pin_memory=True,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=val_sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=num_workers)

    # test_loader = DataLoader(dataset=test_dataset,
    #                         sampler=val_sampler,
    #                         pin_memory=True,
    #                         batch_size=batch_size,
    #                         shuffle=None,
    #                         num_workers=num_workers)
    
    return train_loader, val_loader, train_sampler, val_sampler 




