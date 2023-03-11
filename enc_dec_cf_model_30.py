import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from formal_mask_30 import BartForConditionalGeneration,BartClassificationHead



class Bart(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base") 
        self.projection = nn.Sequential(nn.Linear(768, 768),
                                        nn.GELU(),
                                        #nn.Linear(args.hidden_size, args.hidden_size)
                                        )
        #self.register_buffer("final_logits_bias", torch.zeros((1, self.model.config.vocab_size)))
        self.c_entropy = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768,768)
        self.activation = nn.Tanh()
        self.p_classifier = nn.Linear(768, 2)
       
        
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,labels,overall,shap_attention_mask=None,random_mask=None):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        
        encoder_outputs = encoder(input_ids = input_ids,
                                  attention_mask = attention_mask)
        
        encoder_hidden_states = encoder_outputs[0] # [batch, seq_len, hidden_size]
        pool = self.avg_pool(encoder_hidden_states,attention_mask)
        pool = self.dropout(pool)
        pool = self.activation(pool)
        enc_cf_logits = self.p_classifier(pool)

            
        enc_loss = self.c_entropy(enc_cf_logits,overall)
        
        
        decoder_outputs = decoder(input_ids = decoder_input_ids,
                        attention_mask = decoder_attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=attention_mask,random_mask=False)
        
        decoder_hidden_states = decoder_outputs[0] # [batch, dec_seq_len, hidden_size]
    
        lm_logits = self.model.lm_head(decoder_hidden_states) # + self.final_logits_bias # [batch, dec_seq_len, vocab_size]
        predictions = lm_logits.argmax(dim=2)
    
        criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        nll = criterion(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        pool=self.avg_pool(decoder_hidden_states,decoder_attention_mask)
        pool = self.dropout(pool)
        pool = self.activation(pool)
        dec_cf_logits = self.p_classifier(pool)
            
            
            
        dec_loss = self.c_entropy(dec_cf_logits,overall)
        kl_loss = self.KL(enc_cf_logits,dec_cf_logits)
        
        
        
        
        
        
        
        
        if random_mask:
                
            decoder_outputs = decoder(input_ids = decoder_input_ids,
                                    attention_mask = decoder_attention_mask,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=attention_mask,shap_attention_mask=shap_attention_mask,random_mask=True)
        
            decoder_hidden_states = decoder_outputs[0] # [batch, dec_seq_len, hidden_size]
        
            lm_logits = self.model.lm_head(decoder_hidden_states) 
            predictions = lm_logits.argmax(dim=2)
        
            criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            nll = criterion(lm_logits.view(-1, self.model.config.vocab_size), labels.view(-1))
            
            pool=self.avg_pool(decoder_hidden_states,decoder_attention_mask)
            pool = self.dropout(pool)
            pool = self.activation(pool)
            dec_cf_logits = self.p_classifier(pool)
            
            
            
            dec_loss = self.c_entropy(dec_cf_logits,overall)
            kl_loss = self.KL(enc_cf_logits,dec_cf_logits)
            
            
    

            
            return nll,enc_loss,dec_loss,kl_loss,predictions
        
        
        return nll,enc_loss,dec_loss,kl_loss,predictions
            

    
    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()   
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / (length + 1e-9)

        return avg_hidden
    
    
    
    def KL(self,enc,dec): 
        P_ec=F.softmax(enc,dim=1) 
        P_dec=F.softmax(dec,dim=1)
        kl_div = torch.sum(P_ec *(P_ec.log()-P_dec.log()))
        
        return kl_div