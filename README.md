![header](https://capsule-render.vercel.app/api?type=waving&height=200&fontSize=50&color=gradient&text=KLUE-RE&fontAlignY=26&desc=문장%20내%20개체간%20관계도%20추출(Relation%20Extraction,%20KLUE,%20RE)&descAlignY=49&descSize=23&fontColor=ffffff)

<td style="text-align:center; border-color:white" rowspan="3" width=660>
      <p align="center">
        <img src="/assets/klue-re-img.png" width=600>    
      </p>
</td>  

### <p align="center"><code>KLUE-RE task</code>는</p>
<p align="center">문장의 단어(Entity)에 대한</p>
<p align="center">속성과 관계를 예측하는 NLP Task 입니다. </p>

<table align="center">
  <tr height="8px">
    <td align="center" style="text-align:center;" width="80px">
      <b>Solution 발표자료</b>
    </td>
    <td align="center" style="text-align:center;" width="80px">
      <b>Wrap-up Report</b>
    </td>
  </tr>
  
  <tr height="40px">
    <td align="center" width="150px">
       <a href="https://docs.google.com/presentation/d/14jQBCcV3K_dBka-9P7RxHm9NKQOCS_pwaR7fV0M54hU/edit?usp=sharing"><img src="https://img.shields.io/badge/PPT-%B7472A.svg?&style=flat-square&logo=microsoftpowerpoint&logoColor=white "/></a>
    </td>
    <td align="center" width="150px">
      <a href="/assets/RE-wrapup-report-NLP07.pdf"><img src="https://img.shields.io/badge/PDF-CC2927?style=flat-square&logo=microsoft&logoColor=white">
    </td>
  </tr>
</table>
<p align="center">(↑ 로고를 클릭하면 링크로 이동합니다)</p>
<br>

## 📖 Overview
### 1. 프로젝트 개요
문장 속에서 **단어간에 관계성을 파악**하는 것은 의미나 의도를 해석함에 있어서 많은 도움을 줍니다.

`관계 추출(Relation Extraction)`은 **문장의 단어(Entity)에 대한 속성과 관계를 예측**하는 문제입니다. <br>
`관계 추출`은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. <br> 또한, 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고 중요한 성분을 핵심적으로 파악할 수 있습니다.<br>


- 예시    
    > [`sentence`] <br> 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
    
    > [`subject_entity`]   썬 마이크로시스템즈
    > 
    > 
    > [`object_entity`] 오라클
    > 
    > [`relation`] 단체:별칭 (org:alternate_names)
    > 


### 2. 목표
- RE 데이터셋 속 문장, 단어에 대한 정보를 통해, 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습하는 것을 대회의 목표로 합니다.
- relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 예측하는 것을 목적으로 합니다.



### 3. 데이터셋
- 주어진 Dataset은 Train (32,470, 6), Test (7,765, 6)으로 이루어져 있으며, Valid set은 따로 주어지지 않았다.
- 각 데이터는 id, sentence, subject_entity, object_entity, label, source로 구성되어있다.
- 예시
<table align="center">
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>columns</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>value</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>id</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>0</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>sentence</b>
    </td>
    <td align="center" style="text-align:center;" width="600px">
      <b>〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>subject_entity</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'}</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>object_entity</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>label</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>no_relation</b>
    </td>
  </tr>
  <tr height="8px">
    <td align="center" style="text-align:center;" width="60px">
      <b>source</b>
    </td>
    <td align="center" style="text-align:center;" width="500px">
      <b>wikipedia</b>
    </td>
  </tr>
</table>
<br>
<br>

## 🏅 리더보드 순위
- **public** `2위`→ **private**(최종) `1위`🏅
<p align="center">
  <img width="600px" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/d91e0350-073e-4a54-a0af-d2418cb109d8">
</p>
  <!--<img width="500" alt="image" src="https://github.com/seohyunee22/level2_klue-re_project/assets/152946581/1d226266-5c75-42fb-8e0d-ed9c4fca5632">   -->

<br>
<br>


## 🙆🏻 Members
<table align="center">
  <tr height="8px">
    <td align="center" style="text-align:center;" width="190px">
      <b>공통</b>
    </td>
    <td align="center" style="text-align:center;" width="760px">
      <b>Model 성능 테스트, 하이퍼파라미터 실험, Fine-Tuning</b>
    </td>
  </tr>
</table>
<table align="center">
  <tr height="155px">
    <td align="center" width="170px">
      <a href="https://github.com/seohyunee22"><img src="https://avatars.githubusercontent.com/seohyunee22"/></a>
    </td><td align="center" width="170px">
      <a href="https://github.com/sanggank"><img src="https://avatars.githubusercontent.com/sanggank"/></a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/tmdqor"><img src="https://avatars.githubusercontent.com/tmdqor"/></a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/nachalsa"><img src="https://avatars.githubusercontent.com/nachalsa"/></a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/devBuzz142"><img src="https://avatars.githubusercontent.com/devBuzz142"/></a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/patcasso"><img src="https://avatars.githubusercontent.com/patcasso"/></a>
    </td>
    
  </tr>
  
  <tr height="40px">
    <td align="center" width="170px">
      <a href="https://github.com/seohyunee22">양서현_T6099</a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/sanggank">이상경_T6121</a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/tmdqor">이승백_T6126</a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/nachalsa">이주용_T6137</a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/devBuzz142">정종관_T6157</a>
    </td>
    <td align="center" width="170px">
      <a href="https://github.com/patcasso">정지영_T6158</a>
    </td>
  </tr>

   <tr height="80px">
    <td style="text-align:left;" width="170px">
      - 데이터 클리닝<br>
      - 데이터 전처리<br> (Input Format 변경)<br> 
      - W&B sweep hp 서치
      <!--<b>- W&B Sweep 구현</b><br>-->
    </td>
    <td style="text-align:left;" width="170px">
      - 데이터 분할<br>
      - 하이퍼파라미터 튜닝<br>
    </td>
    <td style="text-align:left;" width="170px">
      - 데이터 분석<br>
      - 데이터 샘플링<br>
    </td>
    <td style="text-align:left;" width="170px">
      - 데이터 분석<br>
      - 데이터 샘플링<br>
      - 앙상블<br>
      - 모델 성능 테스트<br>
    </td>
    <td style="text-align:left;" width="170px">
      - 데이터 전처리<br>(중복값제거)<br>
      - Focal loss 구현<br>
      - 추론 후처리<br>
    </td>
    <td style="text-align:left;" width="170px">
      - 데이터 전처리<br>(Special Token)</b><br>
    </td>
  </tr>
</table>
<br>
<br>

<p align="center">협업관리</p>
<table align="center">
  <tr height="8px">
    <td align="center" style="text-align:center;" width="80px">
      <b>Notion</b>
    </td>
    <td align="center" style="text-align:center;" width="80px">
      <b>Github</b>
    </td>
    <td align="center" style="text-align:center;" width="80px">
      <b>W&B</b>
    </td>
  </tr>
  
  <tr height="40px">
    <td align="center" width="150px">
       <a href="https://www.notion.so/mayy2yy/Level2-KLUE_RE-8ea7f6a4304d428cbfeb7585ca19582f"><img src="https://img.shields.io/badge/notion-%23000000.svg?&style=flat-square&logo=notion&logoColor=white "/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/boostcampaitech6/level2-klue-nlp-07"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white">
    </td>
    <td align="center" style="text-align:center;" width="150px">
      실시간 실험 결과 공유
    </td>
  </tr>
</table>
<p align="center">(↑ 로고를 클릭하면 링크로 이동합니다)</p>
<br>
<br>

## 🛠️ Tech Stack
<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"> <img src="https://img.shields.io/badge/Pytorch Lightening-792EE5?style=flat-square&logo=lightning&logoColor=white"> 
<img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"> 
<img src="https://img.shields.io/badge/scikitlearn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
<br><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"> <img src="https://img.shields.io/badge/notion-%23000000.svg?&style=flat-square&logo=notion&logoColor=white "/>

<!--
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
-->

<br>
<br>

## 💡 프로젝트 수행

<table align="center">
 <tr height="40px">
    <td align="center" style="text-align:center;" width="250px">
      <b>01</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>02</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>03</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>04</b>
    </td>
  </tr>
  <tr height="50px">
    <td align="center" style="text-align:center;" width="250px">
      <b>EDA</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>Data Preprocessing</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>Loss Function</b>
    </td>
    <td align="center" style="text-align:center;" width="250px">
      <b>Postprocessing</b>
    </td>
  </tr>
  <tr height="100px">
    <td align="left" style="text-align:left;" width="260px">
      - Basic Data Information <br>
      - Label, Source 분포<br>
      - 문장 길이 분석<br> 
      - 문장 형태 분석<br> 
    </td>
    <td align="left" style="text-align:left;" width="260px">
      - Data Cleaning<br>(중복 데이터, 한자 및 기타 특수문자 제거) <br>
      - 데이터 분할<br>
      - 데이터 전처리(Input Format 변경)<br>
    </td>
    <td align="left" style="text-align:left;" width="260px">
      - CE Loss <br>
      - Label Smoothing<br>
      - Focal Loss</b><br> 
    </td>
    <td align="left" style="text-align:left;" width="260px">
      - W&B Sweep hp Tunning <br>
      - Data Postprocessing<br>
      - Ensemble<br> 
    </td>
  </tr>
</table>
<br>
🔎 프로젝트 수행과정에 대한 자세한 내용은 <a href="https://www.notion.so/mayy2yy/KLUE-RE-a6e9560421d64b66950186f07839a5f9"><img src="https://img.shields.io/badge/notion-%23000000.svg?&style=for-the-badge&logo=notion&logoColor=white "/></a>(클릭시 이동) 에서 확인하실 수 있습니다.


<br>
<br>

## 🎓 Reference
[1] Wenxuan Zhou, Muhao Chen (2022). An Improved Baseline for Sentence-level Relation Extraction. _arXiv preprint arXiv:2102.01373_.

[2] James Y. Huang, Bangzheng Li, Jiashu Xu, Muhao Chen (2022). Unified Semantic Typing with Meaningful Label Inference. _arXiv preprint arXiv:2205.01826_.

[3] Rafael Müller, Simon Kornblith, Geoffrey Hinton (2020). When Does Label Smoothing Help?. _arXiv preprint arXiv:1906.02629_.

[4] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2018). Focal Loss for Dense Object Detection. _arXiv preprint arXiv:1708.02002_.

---


