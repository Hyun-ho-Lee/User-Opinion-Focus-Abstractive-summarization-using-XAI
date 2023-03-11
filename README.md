# User-Opinion-Focus-Abstractive-summarization-using-XAI


## Process

### 1.Extract Shap Value.


![Encoder Train](https://user-images.githubusercontent.com/76906638/224451254-f65a65a5-cccc-4de6-98b1-fd0b659ffbf4.png)

![예시](https://user-images.githubusercontent.com/76906638/224451231-486651d0-f161-4beb-9df3-773e04cfdf6a.PNG)



### 2.Encoder Hidden state to Cross Attention 

![Random_masking](https://user-images.githubusercontent.com/76906638/224451269-d6d783c9-cf98-40f2-974d-f5c6d35ad56a.png)

Random Masking Module 

    def random_masking(attn_mask,percent):
    
    attn_mask = attn_mask.detach().cpu()

    consecutive_zeros = []
    for i in range(attn_mask.size()[0]):
        nonzero_indices = np.where(attn_mask[i]==0)[0]
        runs = np.split(nonzero_indices, np.where(np.diff(nonzero_indices)!=1)[0]+1)
        if len(runs)>0:
            consecutive_zeros.append(runs)

    for i in range(attn_mask.size()[0]):
        idx = np.random.choice(np.arange(np.prod(attn_mask[i][:consecutive_zeros[i][-1][0]].shape)), 
                                size=int(np.prod(attn_mask[i][:consecutive_zeros[i][-1][0]].shape) * percent),
                                replace=False)
        attn_mask[i, idx] = 1 
        
        
    return attn_mask.to(device)



