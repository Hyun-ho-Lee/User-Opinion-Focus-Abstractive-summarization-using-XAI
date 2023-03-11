# User-Opinion-Focus-Abstractive-summarization-using-XAI

- 본 연구에서는 작성의도에 focus된 텍스트 요약 태스크를 효과적으로 수행하기 위해 sentiment를 고려하는 고도화된 PLM 기반 생성 요약 모델을 제안한다.
- 이는 기존 PLM이 요약문을 생성하는 데에 있어 작성 의도와는 관련되지 않은 문장을 내포하는 경우들이 존재하였으나 본 연구에서는 작성 의도와 관련된 핵심 의도만을 내포한 요약문을 생성하는 것을 연구 목적으로 설정한다. 제안된 방법으로는 PLM의 encoder part를 학습시키는 과정을 통하여 원본의 전체 감성 분포를 파악한 후, Minor Opinion의 일부에 masking을 함으로써 작성 의도에 부합하는 core part인 메이저한 감성에 집중하는 모델을 제안한다.
- 더 나아가 Minor Opinion을 처리함에 있어 Robust한 학습이 가능하도록 고정된 mask가 아니라 어텐션 헤드별로 서로 다른 확률적 masking을 가질 수 있는 novel한 random masking 방식을 함께 제안하였다.


## Process

### 1.Extract Shap Value.


<img src="https://user-images.githubusercontent.com/76906638/224451254-f65a65a5-cccc-4de6-98b1-fd0b659ffbf4.png" width="700" height="400"/>

### 2.Encoder Hidden state to Cross Attention 

<img src="https://user-images.githubusercontent.com/76906638/224451269-d6d783c9-cf98-40f2-974d-f5c6d35ad56a.png" width="600" height="600"/>


#### Random Masking Module 

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



