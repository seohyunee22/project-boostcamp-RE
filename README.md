# level2_klue-re_project
문장 내 개체간 관계 추출(Relation Extraction, KLUE RE)

# 1. Intro

## 1.1. 개요
- `관계 추출(Relation Extraction)`은 단어(Entity) 간의 관계를 예측하는 문제이다.
- 이는 지식 그래프 구축을 위한 핵심 구성 요소로 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요하다.  
- 따라서 주어진 Dataset의 문장 속에서 ,지정된 `두 단어(Entity) 사이의 관계`**와** `단어의 속성`**을 추론하는 모델의 성능을 높이는 것**이 이번 프로젝트의 `목표`이다.
- 문장 속에서 단어 간의 관계성 파악은 의미나 의도를 해석함에 있어서 많은 도움을 준다.
- 요약된 정보를 통한 **QA(Quality Assurance)** 구축과 활용이 가능하며, 이외에도 효율적인 시스템 및 서비스 구성 등이 가능하다.

## 1.2. 리더보드 순위
- public 2위 → private(최종) 1위
  
![](<img width="417" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/8bd86a52-e19d-4602-8998-0c20a69515e1">)

![](<img width="419" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/c96b7d94-beea-4fe8-bd17-6fd055fd9cee">)


# 2. 프로젝트 수행

## 2.1. EDA

### 2.1.1. Dataset
- 주어진 Dataset은 Train (32,470, 6), Test (7,765, 6)으로 이루어져 있으며, Valid set은 따로 주어지지 않았다.
- 각 데이터는 id, sentence, subject_entity, object_entity, label, source로 구성되어있다.
- 예시
|   |   |
|---|---|
|id|0|
|sentence|<Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.|
|subject_entity|{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}|
|object_entity|{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}|
|label|no_relation|
|source|wikipedia|
