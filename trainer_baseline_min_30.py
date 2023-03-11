# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 02:16:29 2022

@author: 이현호
"""

import warnings
warnings.filterwarnings("ignore")


import os
import time
import datasets
import json 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist



from torch.nn.parallel import DistributedDataParallel
from bartdataset_normal_min import get_dist_loader, get_loader
from transformers import AdamW, get_scheduler

from enc_dec_cf_model_30 import Bart
from train_utils import (cal_running_avg_loss, eta, progress_bar,time_since, user_friendly_time)


#%% 

class Trainer():
    def __init__(self, args):
        self.args = args
#       self.tokenizer = args.tokenizer
        self.model_dir = args.model_dir
        
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.val_sampler = None
        
        self.model = None
        self.optimizer = None
        self.rouge = datasets.load_metric("rouge")
        
    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]
        self.args.rank = self.args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                world_size=self.args.world_size, rank=self.args.rank)
        self.model = Bart()


        
        torch.cuda.set_device(self.args.gpu)
        self.model.cuda(self.args.gpu)
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.args.workers = (self.args.workers + ngpus_per_node - 1) // ngpus_per_node
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self.get_data_loader()
        
        param = self.model.parameters()

        
        self.optimizer = AdamW(param, lr=self.args.lr)
        self.num_training_steps = self.args.num_epochs * len(self.train_loader) 

        self.model = DistributedDataParallel(self.model,
                                             device_ids=[self.args.gpu],
                                             find_unused_parameters=True)
        self.lr_scheduler = get_scheduler("linear",
                                          optimizer=self.optimizer,
                                          num_warmup_steps=int(0.1*self.num_training_steps),
                                          num_training_steps=self.num_training_steps)
        
        cudnn.benchmark = True
        
    def get_data_loader(self):
        # TODO change train file to trans_train_file
        train_loader, val_loader, train_sampler, val_sampler = get_dist_loader(batch_size=self.args.batch_size,
                                                                               num_workers=self.args.workers
                                                                               )
        
        return train_loader, val_loader, train_sampler, val_sampler
    
    def train(self, model_path=None):
        running_avg_loss = 0.0
        running_avg_enc_loss = 0.0
        running_avg_dec_loss = 0.0
        batch_nb = len(self.train_loader)
        step = 1
        self.model.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(1, self.args.num_epochs+1):
                start = time.time()
                self.model.train()
                self.train_sampler.set_epoch(epoch)
                for batch_idx, batch in enumerate(self.train_loader, start=1):
                    batch = tuple(v.to(self.args.gpu) for v in batch) 
                    
                    encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,shap_mask,overall,labels  = batch
                    label = labels.squeeze(1)
                    shap_attention_mask=shap_mask.squeeze(1)

                    
                    nll,enc_loss,dec_loss,_,_= self.model(input_ids= encoder_input_ids,
                                                            attention_mask= encoder_attention_mask,
                                                            decoder_input_ids=decoder_input_ids,
                                                            decoder_attention_mask=decoder_attention_mask,
                                                            shap_attention_mask= shap_attention_mask,
                                                            overall = overall,                                                    
                                                            labels = label,random_mask=True)
                    
                    loss = nll+dec_loss
                    loss.backward()  
                       
                                           
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()    
                    self.model.zero_grad()
                    
                    
                    running_avg_loss = cal_running_avg_loss(loss.item(), running_avg_loss)
                    running_avg_enc_loss = cal_running_avg_loss(enc_loss.item(), running_avg_enc_loss)
                    running_avg_dec_loss = cal_running_avg_loss(dec_loss.item(), running_avg_dec_loss)
                    running_avg_summary_loss = cal_running_avg_loss(nll.item(), running_avg_loss)

                    msg = "{}/{} {} - ETA : {} - total loss: {:.4f} - enc loss: {:.4f} - dec loss: {:.4f} - summary loss: {:.4f} ".format(
                        batch_idx, batch_nb,
                        progress_bar(batch_idx, batch_nb),
                        eta(start, batch_idx, batch_nb),
                        running_avg_loss,running_avg_enc_loss,running_avg_dec_loss,running_avg_summary_loss)
                    print(msg,end="\r")
                    step += 1
                    
                # evaluate model on validation set
                if self.args.rank == 0:
                    val_nll,rouge1, rouge2, rougel = self.evaluate(msg)
                    self.save_model(val_nll, epoch)

                    print("Epoch {} took {} - Train total loss: {:.4f} - Train enc loss: {:.4f} - Train dec loss: {:.4f} - Train summary loss: {:.4f} - val NLL: {:.4f} - Rouge1: {:.4f} - Rouge2: {:.4f} - RougeL:{:.4f} "
     .format(epoch,user_friendly_time(time_since(start)),running_avg_loss,running_avg_enc_loss,running_avg_dec_loss,running_avg_summary_loss,val_nll,rouge1,rouge2,rougel))
        
    def evaluate(self, msg):
            val_batch_nb = len(self.val_loader)
            val_losses = []
            self.model.eval()
            for i, batch in enumerate(self.val_loader, start=1):
                
                batch = tuple(v.to(self.args.gpu) for v in batch)
                
                encoder_input_ids,encoder_attention_mask,decoder_input_ids,decoder_attention_mask,overall,labels  = batch
                labels = labels.squeeze(1)
                

                
                with torch.no_grad():
                    nll,_,_,_,predictions= self.model(input_ids= encoder_input_ids,
                                                                            attention_mask= encoder_attention_mask,
                                                                            decoder_input_ids=decoder_input_ids,
                                                                            decoder_attention_mask=decoder_attention_mask,
                                                                            overall= overall,
                                                                            labels = labels,random_mask=False)

                    
            
                self.rouge.add_batch(predictions=predictions, references=labels)     
                msg2 = "{} =>   Evaluating : {}/{}".format(msg, i, val_batch_nb)
                print(msg2, end="\r")
                val_losses.append(nll.item())
                
            score = self.rouge.compute(rouge_types=['rouge1','rouge2','rougeLsum'])
            val_loss = np.mean(val_losses)
            
                
            return val_loss,score['rouge1'][1][2], score['rouge2'][1][2], score['rougeLsum'][1][2]

    
    def save_model(self, loss, epoch):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {"args":self.args,
                "state_dict":model_to_save.state_dict()}
        model_save_path = os.path.join(
            self.model_dir, "{}_{:.4f}.pt".format(epoch, loss))
        torch.save(ckpt, model_save_path)