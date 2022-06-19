# AIFFEL x SOCAR Hackathon
- 팀명: BBC(붕붕케어)
- 주제: **SOCAR 메모 카테고리 자동 분류 챗봇**
- 참여자: 민선아, 서동철, 이정연, 홍성진

<br/>

## 1. 프로젝트 요약
- 이 프로젝트는 모두의 연구소 산하 교육기관 AIFFEL과 카셰어링 기업 SOCAR가 협력해 진행한 해커톤에서 SOCAR 로부터 제공받은 '정비 메모 데이터'를 활용해 진행한 프로젝트입니다.
- 팀 BBC(붕붕케어)는 SOCAR 로부터 제공받은 정비 메모 데이터로부터 고객 상담 데이터를 이용해 `메모 카테고리 자동 분류 챗봇`을 구현해 프로젝트를 진행하였습니다.

  - 다양한 분류 모델을 통해 고객 문의 내역에 대한 카테고리를 분류하고,
  - 클릭형 시나리오 챗봇을 구축해 웹에서 구현하였습니다.

<img width="549" alt="image" src="https://user-images.githubusercontent.com/74005372/174451711-234c18d2-6067-40a9-870f-b81936332073.png">


<br/>

## 2. 프로젝트 소개

### 배경
- 불만 접수에 대한 고객과 상담사 간 대화 과정에서 불필요한 시간, 비용, 감정이 소모된다는 점 인지
- 상담원이 직접 카테고리를 수작업으로 분류한다는 문제점 파악
- 반복적인 챗봇 서비스로 인한 고객의 불만 증대

### EDA
- 총 99,180개의 데이터 중 고유 memo 9,450개를 추출한 뒤 최종적으로 `문의 내역` 데이터 5,836개 데이터 추출
- 형태소 분석을 통한 띄어쓰기
  - Okt, Mecab을 이용해 재띄어쓰기 시도
- 불용어 제거
  - SubwordTextEncoder를 이용해 생성한 단어장으로 불필요한 stopwords 제거
- Class Imbalance 해결
  - RandomOverSampler를 통해 총 21개 sub-category 중 가장 많은 class의 data의 수만큼 augmentation 시킴

### Modeling
1. Sequential (RNN)
- multi class classification을 위한 dense layer 변경
- pretrained 되지 않고 충분한 데이터셋이 확보되지 않은 이유로 좋은 성능을 보이지 못함

```
subtype_predict('내비게이션이 안 됩니다.') # 내비게이션
# 블랙박스
```

<img width="488" alt="image" src="https://user-images.githubusercontent.com/74005372/174451053-26724858-deac-40ed-af3d-f302ea9da4d9.png">

<br/>

2. Transformer
- 생성 모델을 이용해 분류 문제를 해결하기 위해 분류를 위한 완전 연결층 추가

```
sentence_generation('블랙박스 확인 요청 부탁드립니다.') # 사고조사
# 블랙박스
```

<img width="450" alt="image" src="https://user-images.githubusercontent.com/74005372/174451018-5c59cab0-717f-4a6d-9d24-80d5d4f2f97a.png">

<br/>

3. BERT
- pre-trained된 bert-base-multilingual-cased 모델을 사용해 분류를 위한 layer를 추가로 쌓아 성능 개선

```
input the text: 블랙박스 확인 요청 부탁드립니다.
# 사고조사 98.1517
```

<img width="512" alt="image" src="https://user-images.githubusercontent.com/74005372/174451057-f82f4f4d-c2c5-4f27-9c03-78c19b0bca2d.png">




<br/>

### 장고 웹 구현
<img width="1336" alt="image" src="https://user-images.githubusercontent.com/74005372/174451186-359420ca-0720-4987-8b71-0ef37563e9ac.png">
<img width="1336" alt="image" src="https://user-images.githubusercontent.com/74005372/174452155-8d93ead0-afc4-4909-b6af-2c5e83c1b1a2.png">


<br/>

## 3. 향후 연구 과제
- 세분화된 카테고리를 통해 카테고리 재구성 및 re-category
- 다양한 text data augmentation 기법 적용
- dialogue state tracking을 이용한 대화형 문의 상담 챗봇 구현
- 감정분석을 적용한 사용자 맞춤형 페르소나 챗봇 구축

<br/>

## 4. 회고
- 팀원 모두 처음 도전한 nlp 프로젝트였음에도 흥미를 갖고 다양한 분석을 시도할 수 있었다.
- 성능 하락의 큰 원인이었던 클래스 불균형을 해결하기 위해 text data의 다양한 augmentation 기법을 찾아보았고, 현재도 꾸준히 연구되고 있는 영역이라는 점에 큰 매력을 느낄 수 있었으며 다양한 방법론을 찾아 도전해 볼 수 있었다.
- raw data의 정제 및 전처리 과정이 의미있었다.
- 다중 클래스 분류를 위해 3가지 모델을 사용해 보았고, 각 모델들의 특징을 비교하며 성능 개선을 시도할 수 있었다.
