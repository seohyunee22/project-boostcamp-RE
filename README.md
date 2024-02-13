
---
# level2_klue-re_project
문장 내 개체간 관계 추출(Relation Extraction, KLUE RE)

# Intro

## 대회 개요
- 한국어 모델의 성능을 평가하기 위한 데이터셋인 KLUE(Korean Language Understanding Evaluation)의 8가지의 대표적인 task중 하나인 관계 추출(RE, Relation Extraction)을 수행하는 모델 제작
- `관계 추출(Relation Extraction)`은 단어(Entity) 간의 관계를 예측하는 문제이다.
- 이는 지식 그래프 구축을 위한 핵심 구성 요소로 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다.  
- 따라서 주어진 Dataset의 문장 속에서 ,지정된 `두 단어(Entity) 사이의 관계`**와** `단어의 속성`**을 추론하는 모델의 성능을 높이는 것**이 이번 프로젝트의 `목표`이다.
- 문장 속에서 단어 간의 관계성 파악은 의미나 의도를 해석함에 있어서 많은 도움을 준다.
- 요약된 정보를 통한 **QA(Quality Assurance)** 구축과 활용이 가능하며, 이외에도 효율적인 시스템 및 서비스 구성 등이 가능하다.

## 리더보드 순위
- **public** `2위`→ **private**(최종) `1위`🏅
  
  <img width="700" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/1d226266-5c75-42fb-8e0d-ed9c4fca5632">
  

- private(최종) `1위`
  
  <img width="700" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/d91e0350-073e-4a54-a0af-d2418cb109d8">



## 프로젝트 역할 및 구성


## 협업 툴

- Notion
    - 공지 및 project kanban board 운용
- Github
    - Git, PR, Action을 통한 자동화
- W&B
    - 실시간 실험 내용 및 결과 공유

## Skills

- Pytorch
- PyTorch Lightning
- HuggingFace
- Pandas, Numpy
- Scikit-Learn

## Directory
```
📦 level2_klue-nlp-07
├─ train_code
│  ├─ train.py
│  ├─ kfold_train.py
│  └─ sweep_train.py
└─ utils
│  ├─ preprocessing.py
│  ├─ aeda.py
│  ├─ dict_label_to_num.pkl
│  ├─ dict_num_to_label.pkl
│  ├─ losses.py
│  └─ metrics.py
├─ README.md
├─ load_data.py
├─ inference.py
└─ requirements.txt
```

## 프로젝트 수행

## EDA

### 1.  Dataset

- 주어진 Dataset은 Train (32,470, 6), Test (7,765, 6)으로 이루어져 있으며, Valid set은 따로 주어지지 않았다.
- 각 데이터는 id, sentence, subject_entity, object_entity, label, source로 구성되어있다.
- 예시

| id | 0 |
| --- | --- |
| sentence | 〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다. |
| subject_entity | {'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'} |
| object_entity | {'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'} |
| label | no_relation |
| source | wikipedia |

### 2.  Label, Source 분포
<img width="450" alt="image" src="https://lh7-us.googleusercontent.com/UNFeBLA2Px8oLvzPWeoXoNx0Az9Um5T7VmXP3_k2kequUjKXxD3hLYcsmUCuTRRVcFA2KhL2oaI8VY8T2UYNnLXPrr6MRMptLbZkFNEuDDtNKE6kKtgx-TIrdZewy8HKJZM109jJOzrd7M60j2kyFMNs-A=s2048"> <img width="350" alt="image" src="https://lh7-us.googleusercontent.com/oSyk_fdt3m7ZTGJlnFaxXTLKBc44H-RZ6PWk2BL-IYenRjPYh6uLhJ1fNdABTzYR0S1knr_8jZFn26m4D3F0msyblG3dMLeZt38DLS6QkL4bHpvl8EHr1eODuoXcUv38ksiztVZoij4fkxdJOrPZ5m6JfQ=s2048">



- Train 데이터 : (32470, 6),  Test 데이터 : (7765, 6)
- 가장 많은 Label(no_relation) 9,534 개, 가장 적은 Label(per:place_of_death)가 40개로 분포가 Imbalanced 한 분포를 가진다.

## Data Preprocessing

### 데이터 클리닝(Data Cleaning)

### 1.  중복 데이터 제거

| id | sentence | subject | object | label |
| --- | --- | --- | --- | --- |
| 6749 | 대한항공은 5일 조양호 회장의 3자녀가 보유한 싸이버스카이… | 대한항공, ORG | 조양호, PER | no_relation |
| 12829 |  |  |  | org:top_members/employees |
| 8364 | 배우 김병철 씨가 연기하는 정복동은 천리마마트를 망하게 … | 정복동, PER | 김병철, PER | no_relation |
| 32299 |  |  |  | per:alternate_names |
| 11511 | 영화 \'버즈 오브 프레이\'는 배트맨이 없는 고담시에서 할리퀸… | 배트맨, PER | 고담시, LOC | per:place_of_residence |
| 22258 |  |  |  | no_relation |
| 277, 10202 | 이날 프로그램 공개에서는 전북영산작법보존회와 김명신‧정상희… | 강태환, PER | 색소폰, POH | no_relation |
| 3296 |  |  |  | per:title |
| 4212 | 한편 전라남도는 최근 확진자가 발생한 순천시와 여수시에… | 전라남도, ORG | 여수시, LOC | org:members |
| 25094 |  |  |  | org:place_of_headquarters |
- sentence / subject / object 가 전부 동일한 데이터를 조회 (중복 데이터 조회)
- 그 중 Label이 잘못된 데이터 5개를 Drop 하였다.

### 2.  한자 및 기타 특수 문자 제거

- 주어진 sentence 32,469개 중 한자를 포함하는 문장 수가 총 2,308개였다.
- Entity word에 한자가 포함되어 있는 경우를 제외한 모든 한자 제거
    - 한자는 대부분 `[UNK]` **토큰**으로 처리되어, Entity word 자체에 한자가 포함 되어있는 단어를 제외한 sentence 내 모든 한자를 제거하면 학습에 도움이 될 것이라 예상하였다.
    - 예시
        
        
        | sentence | sub entity word | obj entity word |
        | --- | --- | --- |
        | 김동성( 金東聖 (삭제), 1980년 2월 9일 ~)은 대한민국의 … | 김동성 | 1980년 2월 9일 |
        | 신 안동 김씨가 권력의 기반을 잡은 것은 광해조 때 도정을 지낸 김극효(金克孝)가 … | 신 안동 김씨 | 김극효(金克孝) |
- 문장 부호 제거
    - `[UNK]` 토큰으로 처리되는 특수 문자
        
        ```
        ['↔', '–', '€', 'Ⓐ', '☎', '。', '˘', '＇', '́', '⸱', '∞', '･', '⋅', '⟪', '⟫', '～',
         '£','°', '˹', '˼', '▴', '（', '）', '·', '☏', '⁺', '®', '、', '？', '／', '़', 'ी',
         'ु', '्', 'ा', '㈔', '▵', '«', '»', '′', '₫', '㎿', '⌜', '⌟', '±']
        ```
        
    - tokenizer vocab에 이미 존재하는 문장 부호를 제외하고 **[UNK]**로 처리되는 다양한 문장부호를 제거해 모델의 성능 향상을 꾀했다.
    - 예시
        
        
        | sentence | sub entity word | obj entity word |
        | --- | --- | --- |
        | 량치차오(梁啓超)가 “소설계 혁명(小說界革命)”을 제기하고 일본에서  ≪ (삭제) 신소설 ≫ (삭제) 잡지를 … | 량치차오 | 梁啓超 |
    - 결과
        - 최종적으로 [UNK] token을 포함된 문장을 **총 171개 까지 줄일 수** 있으나, 오히려 성능이 떨어지는 결과가 나와 채택하지 않았다.
        

### 데이터 분할(Data Split)
<img width="700" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/c8d9c8dc-68f4-4260-a194-2c7f966ea3bb">

- 기존 Dataset 분할
    - Stratified Shuffle Split 과 sklearn.train_test_split을 사용하여 Label 비율을 Train : Valid = 8 : 2 유지하는 방향으로 데이터셋 분할
- 문제 인식
    - Valid 분할 시 validation f1 score는 높지만 리더 보드 제출 시 public f1 score와 차이가 상당하였고, 이전보다는 score가 향상 했으나 눈에 띠는 성능 향상은 아니었다.
    - Label 기준으로 Sampling할 경우, 특정 Label의 데이터 개수가 매우 적어 관계가 Valid set에 반영되지 않는 문제가 발생하기 때문이다. 
    e.g. `(NOH, subject_type) pair` : (NOH, ORG), (NOH, PER)
- 반영한 분할 방법
    - object type(6 종류) - subject type(2 종류)로 이루어진 총 12개의 부분에서 특정 개수만큼 균등하게 뽑아 valid dataset 확보
    

## 초기 실험

### 모델 선정

- **`klue/RoBERTa-large` : 최종 모델 선정**
- klue/bert-base : 빠른 실험을 위해 병행 사용
- snunlp/KR-SBERT-V40K-klueNLI-augSTS : 평가 지표가 낮게 나와 미 사용
- paust/pko-t5-base : 실험 속도가 너무 느려 미 사용
- paust/pko-t5-large : 실험 속도가 너무 느려 미 사용

### 데이터 증강 (EDA, AEDA)

- BERT 와 같이 Masking 사용하는 모델에는 효과적이지 않다. (논문)
- 실제 적용 시 성능 향상이 뚜렷하게 드러나지 않았다.

### n**o-relation 을 우선적으로 판별하는 이진 분류 모델**

- no-relation / relation으로 1차 분류
- relation data에 대하여 추가로 multi-class 분류
- 결과
    - 단일 모델에 비해 성능 향상을 확인하지 못함
- 김성현 마스터님 피드백
    - 이진 분류 후 세부 분류는 항상 성능이 떨어지는 경우가 많다. 에러 전파 때문일 거 같다.

### Token 활용

- token을 통해 Entity Type을 더 명확히 기술해줄 때 성능이 향상되는 것을 확인하였다.

## Input Format 변경

> (3강-실습-1) `Special Token 추가 방법론`과 논문 ≪[Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/pdf/2205.01826v1.pdf)≫ 와 ≪[Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/pdf/2205.01826v1.pdf)≫을 참조해 **Input Format 을 변경**하였다.
> 

### 구조도

- **sentence = 조지 해리슨이 쓰고 비틀즈가 앨범에 담은 노래다**
- 전체적인 구조도
<img width="700" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/41722b85-4a71-4443-9abb-40ccea347aa6">

### A.  Entity Marker(**Entity Masked Special Token)**

- 착안 : 문장 안의 두 단어의 관계 만을 추론한다면 단어가 어떤 형태이든 상관 없지 않을까?
- 가설 : 두 Entity word 대신 각각 <S-type>, <O-type> Special Tokens로 Masked 하여 학습의 input format을 변경해 학습한다면 모델의 성능이 향상될 것이다.
- 예시
    
    > `기존`    [CLS] 비틀즈 [SEP] 조지 해리슨 [SEP] Sentence [SEP] [PAD]
    `변경`    [CLS] [S:ORG] [SEP] [O: PER] [SEP] [O: PER] 이 쓰고 [S:ORG] 가 앨범에 담은 노래
                 다[SEP]
    > 
    - (baseline) 66.9683 → 67.22 로 성능이 소폭 향상
- 후에 <S-type> 와 <O-type>에 해당하는 Special token을 추가하고 Embedding Layer 을 추가하였다.
    
    ```python
    # Special token 추가
    if sp_token_list is not None :
    	tokenizer.add_special_tokens({'additional_special_tokens':sp_token_list})
    
    # Embedding Layer 추가
    model.resize_token_embeddings(len(tokenizer))
    ```
    

### B.  Typed Entity Marker `w/Special Tokens`

- 착안 : (3강-실습-1) BERT 언어 모델 소개’ 에서 제시된 Special Token 추가 방법론에서 착안
- 가설 : `[S:Type]` SUBJECT `[/S:Type]` , `[O:Type]` OBJECT `[/O:Type]` 처럼 Entity Type 정보를 스페셜 토큰으로 추가하면 모델의 성능이 향상될 것이다.
- 예시
    
    > `기존`    [CLS] 비틀즈 [SEP] 조지 해리슨 [SEP] Sentence [SEP] [PAD]
    > 
    > 
    > `변경`    [CLS] `[S:ORG]` 비틀즈 `[/S:ORG]` [SEP] `[O: PER]` 조지 해리슨 `[/O: PER]` 
    >              [SEP] `O: PER]` 조지 해리슨 `[/O:PER]`이 쓰고 `[S:ORG]` 비틀즈 `[/S:ORG]`가 앨범에 
    >              담은 노래다[SEP]
    > 
- 결과
    - 리더보드 F1-score 기준 **62.5111 → 65.4545** 으로 성능 향상 (klue/bert-base)
    

### C.  Typed Entity Marker `w/Punctuation`

- 착안 : (3강-실습-1) Special Token 추가 방법론 
          + ≪[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/pdf/2102.01373.pdf)≫ 논문
- 가설
    - Special Token이 필요한 처리는 새로 학습이 필요하다.
    - [CLS] `@*Type*` SUBJECT `@` [SEP] `#^Type^` OBJECT `#` [SEP] + Sentence 와 같은 형태로, 
    이미 기 학습된 특수 문자를 사용하여 Entity의 정보를 표현하면 성능이 향상될 것이다.
    - 즉, Entity 정보를 special token 말고 특수 문자로 치환해보자.
- 예시
    
    > [CLS] `@*ORG*` 비틀즈 `@` [SEP] `#^PER^` 조지 해리슨 `#` [SEP]
    > 
    > 
    > `#^PER^` 조지 해리슨 `#`이 쓰고 `@*ORG*` 비틀즈 `@`가 앨범에 담은 노래다[SEP]
    > 
- 결과 : 리더보드 제출 F1 score 기준 **66.9683 → 67.22** 으로 성능 향상

### D.  Add Semantic Typing Query

- 착안 : Bert와 같은 사전 학습 모델의 NSP(Next Sentence Prediction) 학습 방식 
          + ≪[Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/pdf/2205.01826v1.pdf)≫ 논문
- 가설 : 두 Entity word 대신 두 단어의 관계를 설명하는 Query 를 생성하여, 성능이 향상된 
          Typed Entity Marker `w/Punctuation` sentence와 전달해준다면 성능이 향상될 것이다.
- 예시 (순서가 다름)
    - `w/Punctuation` Sentence : #^PER^조지 해리슨#이 쓰고 @*ORG*비틀즈가@ 앨범에 담은 노래다.
    1. Semantic Query + Typed Entity Marker w/Punctuation
        
        > [CLS] `@*ORG*` **비틀즈** `@` **와** `#^PER^` **조지 해리슨** `#` **의 관계는 ORG와 PER이다.**
        > 
        > 
        > [SEP] `w/Punctuation` Sentence [SEP]
        > 
    2. Typed Entity Marker w/Punctuation + Semantic Query
        
        > [CLS] `w/Punctuation`Sentence [SEP]
        > 
        > 
        > `@*ORG*` **비틀즈** `@` **와** `#^PER^` **조지 해리슨** `#` **의 관계는 ORG와 PER이다.** [SEP]
        > 
- 결과 :
    - 순서가 변경된 Input Format은 기존 순서보다 성능이 낮다.
    - 리더보드 제출 F1 score 기준 **69.5031 → 73.7805(73.0421)**으로 성능 향상 ****
    

### F. Add Semantic Typing Query (Korean)

- 착안 : 이미 한국어 Embedding이 되어있는 모델을 사용하는데 Type을 영어로 명시하는 것 보단 
          한국어로 변경하면 더 낫지 않을까?
- 가설 : Type을 한국어로 바꿔 학습하면 모델의 성능이 향상될 것이다.
- 예시
    
    > #^**PER**^#조지 해리슨#과 @***ORG***비틀즈@의 관계는 **PER**과 **ORG**이다.
    →  `@*조직*` **비틀즈** `@` **와** `#^사람^` **조지 해리슨** `#` **의 관계는 조직과 사람이다.**
    > 
- 결과
    - 리더보드 제출 F1 score 기준 **73.7805 → 75.4402** 으로 성능 향상 ****
    - 최종 Private F1 score는 English 버전이 더 높았다.
    

### F.  Source Marker w/Special Tokens

- 착안 : EDA 과정에서, 데이터 source에 따라 라벨 분포에 경향성이 있는 것을 보고 착안
<img width="500" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/f25ae02e-39f6-4848-b620-145f779175f4">

- 가설 : [CLS] SUBJECT [SEP] OBJECT ****[SEP] `[Source]` ****[SEP] + Sentence 와 같은 형태로,
          Input 데이터에 Source 스페셜 토큰을 추가하면 모델의 성능이 향상될 것이다.
- 예시
    
    > [CLS] 비틀즈 [SEP] 조지 해리슨 ****[SEP] `[wikipedia]` [SEP] + Sentence
    > 
- 결과
    - 리더보드 제출 F1 score 기준 약 **69.5031 → 68.8676** 으로 오히려 **성능 하락**
    - source가 policy_briefing 에 해당하는 데이터의 개수가 너무 적고, Train 데이터의 source 별 데이터 분포가 Test 데이터와는 다를 수 있기 때문으로 추정
    

### → 최종 Input Format

- Typed Entity Marker `w/Punctuation`(ENG, KOR) + Add **Semantic Typing Query**
- 예시
    1. punctuation_ENG + Semantic Typing Query
    
    > [CLS] `@*ORG*` **비틀즈** `@` **와** `#^PER^` **조지 해리슨** `#` **의 관계는 ORG와 PER이다.**
    > 
    > 
    > [SEP] `#^PER^` 조지 해리슨 `#`이 쓰고 `@*ORG*` 비틀즈 `@` 가 앨범에 담은 노래다. [SEP]
    > 
    1. punctuation_KOR + + Semantic Typing Query
    
    > [CLS] `@*조직*` **비틀즈** `@` **와** `#^사람^` **조지 해리슨** `#` **의 관계는 조직과 사람이다.**
    > 
    > 
    > [SEP] `#^사람^` 조지 해리슨 `#`이 쓰고 `@*조직*` 비틀즈 `@` 가 앨범에 담은 노래다. [SEP]
    > 

---

## Loss Function

### CE Loss (CrossEntropyLoss)

- 다중 클래스 분류에 주로 사용되는 loss로, baseline에 설정되어 있던 default loss이다.

### Label Smoothing

- Dataset에 특정 Label이 많은 경우 불균형 해소를 위한 방법이다.
- Label을 0 or 1 이 아니라 smooth 하게 부여해 모델의 overfitting을 막아주고 regularization 효과를 기대할 수 있다.
- 참고 논문 - ≪[When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629)≫
- 결과 : CE Loss 대비 큰 성능 향상은 없었다.

### Focal Loss

- 클래스 불균형 문제를 해결하기 위해 오분류된 클래스나 분류가 어려운 클래스에 더 높은 가중치를 부여한다.
- 참고 논문 - ≪[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)≫
- 결과 : 리더보드 제출 F1 score 기준 **73.7805 → 73.8524**으로 성능 향상

## Hyper Parameters Tunning

- W&B Sweep을 활용하여 다양한 요소들을 변경하여 최적의 Hyper-Parameter를 찾으려 했다.
    
    ```python
    sweep_config = {
          "name": "klue-nlp7-lv2",
          "method": "bayes",
          "metric": {"goal": "maximize", "name": "eval/micro f1 score"},
          "parameters": {
              "preprocessing_mode": {"values": [ "punct_kr", "punct_eng"]},#, "entity_masking", "default"
              "num_train_epochs": {"min": 3, "max": 6},
              "learning_rate": {"min": 15e-6, "max": 5e-5},
    					"loss" : {'values':["CeLoss","FocalLoss"]}
              "per_device_train_batch_size": {"values":[8, 16, 32]},
              "per_device_eval_batch_size": {"values":[8, 16, 32]},
          }
      }
    ```
    

### Learning Rate
<img width="500" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/61ff9541-f2a4-436b-b9c3-17280d883d6a">

- 실험 : sweep을 활용한 다양한 learning rate 학습 시도 및 최적의 learning rate 탐색
- 결과 : 리더보드 F1-score 기준 **70.2360 → 73.83748** 으로 성능 향상

### Random Seed

<img width="500" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/3d8cbc12-9481-4db6-b221-00f14d3e6679">

- 착안 : 첫 step의 train/loss가 낮으면 일반적으로 성능이 높았다.
- 가설 : seed의 변경으로 처음 step의 train/loss 값을 낮추면 성능이 올라가지 않을까?
- 결과 :  리더보드 F1-score 기준 **74.3179 →  74.8019** 으로 성능 향상

---

## Data Postprocessing

### Subject Type - Label Head 일치

| id | sentence | sub_word | sub_type | pred |
| --- | --- | --- | --- | --- |
| 10 | 실제로 틱톡의 ‘극한반전 챌린지’는 제2의 이병헌 감독을 … | 틱톡 | PER | org:top_members/employees |
| 201 | 춘신군은 아래에 식객(食客) 3천 인을 모아 거느리고 있었는데, … | 춘신군 | ORG | per:employee_of |
| 235 | 2008년, 브라이언 싱어의 스릴러 작전명 발키리 (Valkyrie) 에서 … | Christian Berkel | ORG | per:origin |
| … |  |  |  |  |
| 7106 | 채성의 대표는 강진읍 평동리에 위치한 토목설계를 주 업종 … | 채성 | ORG | per:title |
| 7111 | 이후 단부(段部) 토벌을 주도하며 단감(段龕)을 붙잡는 데 성공하며 단부를 멸망시켰... | 段龕 | PER | org:alternate_names |
| 7637 | 두 메클렌부르크 대공국은 서슬라브족의 혈통이나 게르만화 … | Obodriten | ORG | per:alternate_names |
- 착안
    - 원본 train.csv 에 subject type과 label head(org:, per:)가 일치하지 않는 경우 : `4` / 32,470
    - baseline 코드로 실험 결과, test_data.csv로 inference 후 나온 pred_label의 `57` / 7,765 의 data에서 subject type과 pred_label_head가 일치하지 않음을 확인하였다.
- 가설 : subject type과 Label head를 일치 시키면 모델 성능이 향상될 것이다.
- 실험 : 최종 예측 label을 결정할 때, subject type에 맞지 않은 label은 고려하지 않음
- 결과 : 0.6 % 성능 향상 (67.8029 → 68.2653)

## Ensemble

- 최대한 다른 전처리 방식과 seed의 모델들을 여러 조합으로 Soft voting Ensemble 하였다.
    
- 단일 모델 최고점 75.4402 → `76.9665` 점 기록
    - klue/RoBERTa-large 고정
        
        > 1.   lr = 18e-06 / seed = a / punctuation_kor / epoch = 4 / CE Loss / …
        > 2.   lr = 18e-06 / seed = a / punctuation_eng / epoch = 4 / Focal Loss / …
        > 3.   lr = 1.5e-05 / seed = b / punctuation_kor / epoch = 4 / CE Loss / … 
        > 4.   lr = 1.5e-05 / seed = c / punctuation_kor / epoch = 4 / CE Loss / …
                                                                      …
        > 

---

## Reference
[1] Wenxuan Zhou, Muhao Chen (2022). An Improved Baseline for Sentence-level Relation Extraction. _arXiv preprint arXiv:2102.01373_.

[2] James Y. Huang, Bangzheng Li, Jiashu Xu, Muhao Chen (2022). Unified Semantic Typing with Meaningful Label Inference. _arXiv preprint arXiv:2205.01826_.

[3] Rafael Müller, Simon Kornblith, Geoffrey Hinton (2020). When Does Label Smoothing Help?. _arXiv preprint arXiv:1906.02629_.

[4] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2018). Focal Loss for Dense Object Detection. _arXiv preprint arXiv:1708.02002_.


---

## 회고

- 개인적인 시도 AEDA
- LabelSmoothingLoss
