from distutils.log import error
from xml.etree.ElementPath import prepare_predicate
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from konlpy.tag import Okt

# 하이퍼파라미터
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
DROPOUT = 0.1 # 드롭아웃의 비율

df = pd.read_csv('static/no_nan_qna_set.csv', index_col=0)

# okt = Okt()
# okt 형태소 분석기로 토큰화한 문의내역, subtype 리스트 저장
inquiry_token = []

# for i in range(len(df)):
#   inquiry_token.append(okt.morphs(df['inquiry'][i]))

# okt 형태소 분석기를 통해 분리된 기준으로 띄어쓰기 하기
# 방법 1
# for sentence in inquiry_token[:2]: # ['브레이크', '가', '많이', '밀림', '차량', '변경', '요청']
#   temp = ''
#   for word in sentence: # 브레이크 가 많이 밀림 차량 변경 요청
#     temp += word + ' '
#   temp = temp.rstrip()
# 방법 2 -> 채택
# questions_okt = []
# for sentence in inquiry_token:
#   temp = ' '.join(sentence)
#   questions_okt.append(temp)

########### 단어장 만들기 ###########
import tensorflow_datasets as tfds
print("살짝 오래 걸릴 수 있어요. 스트레칭 한 번 해볼까요? 👐")

questions = list(df['inquiry'])
# 질문과 답변 데이터셋에 대해서 Vocabulary 생성. (Tensorflow 2.3.0 이상) (클라우드는 2.4 입니다)
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions, target_vocab_size=2**13) # questions만!
# 시작 토큰과 종료 토큰에 고유한 정수를 부여합니다.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
# 시작 토큰과 종료 토큰을 고려하여 +2를 하여 단어장의 크기를 산정합니다.
VOCAB_SIZE = tokenizer.vocab_size + 2
tokenizer_questions = []
for i in range(len(questions)):
  tokenizer_questions.append(tokenizer.encode(questions[i]))

# # 새롭게 word_to_index 생성!!!!!
# # X 자리에 tokenizer_questions 넣기
# from collections import Counter

# # def make_word_to_index(X):
# # words = np.concatenate(tokenizer.subwords).tolist()
# counter = Counter(tokenizer.subwords) # 각 요소의 개수 다루고 싶을 때
# counter = counter.most_common(2774-4) # 빈도순으로 높은 9996개 리스트 안의 튜플로 반환
# vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter] # vocab 저장
# word_to_index = {word: index for index, word in enumerate(vocab)}
# index_to_word = {index: word for word, index in word_to_index.items()}

########### 불용어 제거 ###########
import re
new_memo = []
for i in range(len(df)):
    sentence = re.split("[\n | / | ) | , | _ | 재인입 | 인입 | 문의 | 요청 | 확인]", df['inquiry'][i]) # 그냥 split은 여러개 안돼서, re.split 사용
    temp = ' '.join(sentence)
    temp = re.sub(r"\s+", " ", temp).strip()
    new_memo.append(temp)
df['inquiry'] = new_memo

########### 포지셔널 인코딩 레이어 ###########
class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)
  
  def get_config(self):
      config = super().get_config()
      config.update({
          "pos_encoding": self.pos_encoding,
      })
      return config

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    # 각도 배열 생성
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    # 배열의 짝수 인덱스에는 sin 함수 적용
    sines = tf.math.sin(angle_rads[:, 0::2])
    # 배열의 홀수 인덱스에는 cosine 함수 적용
    cosines = tf.math.cos(angle_rads[:, 1::2])

    # sin과 cosine이 교차되도록 재배열
    pos_encoding = tf.stack([sines, cosines], axis=0)
    pos_encoding = tf.transpose(pos_encoding,[1, 2, 0]) 
    pos_encoding = tf.reshape(pos_encoding, [position, d_model])

    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

########### 스케일드 닷 프로덕트 어텐션 함수 ###########
def scaled_dot_product_attention(query, key, value, mask):
  # 어텐션 가중치는 Q와 K의 닷 프로덕트
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # 가중치를 정규화
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # 패딩에 마스크 추가
  if mask is not None:
    logits += (mask * -1e9)

  # softmax적용
  attention_weights = tf.nn.softmax(logits, axis=-1)

  # 최종 어텐션은 가중치와 V의 닷 프로덕트
  output = tf.matmul(attention_weights, value)
  return output

########### 멀티헤드 어텐션 ###########
# 내부적으로는 스케일드 닷 프로덕트 어텐션함수를 불러온다
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "num_heads": self.num_heads,
        "d_model": self.d_model,
        "depth": self.depth,
        "query_dense": self.query_dense,
        "key_dense": self.key_dense,
        "value_dense": self.value_dense,
        "dense": self.dense,
    })
    return config

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # Q, K, V에 각각 Dense를 적용합니다
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # 병렬 연산을 위한 머리를 여러 개 만듭니다
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # 스케일드 닷 프로덕트 어텐션 함수
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # 어텐션 연산 후에 각 결과를 다시 연결(concatenate)합니다
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # 최종 결과에도 Dense를 한 번 더 적용합니다
    outputs = self.dense(concat_attention)

    return outputs

########### 패딩 마스킹 함수 ###########
def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

########### 인코더 레이어 ###########
# 인코더 하나의 레이어를 함수로 구현.
# 이 하나의 레이어 안에는 두 개의 서브 레이어가 존재합니다.
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

  # 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 첫 번째 서브 레이어 : 멀티 헤드 어텐션 수행 (셀프 어텐션)
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })

  # 어텐션의 결과는 Dropout과 Layer Normalization이라는 훈련을 돕는 테크닉을 수행
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  # 두 번째 서브 레이어 : 2개의 완전연결층
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 완전연결층의 결과는 Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)
    

########### 인코더 함수 ###########
def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 임베딩 레이어
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

  # 포지셔널 인코딩
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # num_layers만큼 쌓아올린 인코더의 층.
  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


########### 디코더 레이어 ###########
# 디코더 하나의 레이어를 함수로 구현.
# 이 하나의 레이어 안에는 세 개의 서브 레이어가 존재합니다.
def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 첫 번째 서브 레이어 : 멀티 헤드 어텐션 수행 (셀프 어텐션)
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })

  # 멀티 헤드 어텐션의 결과는 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  # 두 번째 서브 레이어 : 마스크드 멀티 헤드 어텐션 수행 (인코더-디코더 어텐션)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })

  # 마스크드 멀티 헤드 어텐션의 결과는
  # Dropout과 LayerNormalization이라는 훈련을 돕는 테크닉을 수행
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  # 세 번째 서브 레이어 : 2개의 완전연결층
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 완전연결층의 결과는 Dropout과 LayerNormalization 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


########### 디코더 함수 ###########
def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')

  # 패딩 마스크
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  # 임베딩 레이어
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))

  # 포지셔널 인코딩
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  # Dropout이라는 훈련을 돕는 테크닉을 수행
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

# original
questions = list(df['inquiry'])
answers = list(df['sub_type'])

total_data_text = tokenizer_questions # 위에서 불용어 제거한 questions를 encoding 한 것
# 텍스트 데이터 문장길이의 리스트 생성

num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens)
maxlen = int(max_tokens)

# 샘플의 최대 허용 길이 또는 패딩 후의 최종 길이
MAX_LENGTH = maxlen

# 정수 인코딩, 최대 길이를 초과하는 샘플 제거, 패딩
def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 25 이하인 경우에만 데이터셋으로 허용
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # 최대 길이 26으로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs

# original questions, answer 이용
questions, answers = tokenize_and_filter(questions, answers)

BATCH_SIZE = 64
BUFFER_SIZE = 2774

# 디코더는 이전의 target을 다음의 input으로 사용합니다.
# 이에 따라 outputs에서는 START_TOKEN을 제거하겠습니다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

########### 트랜스포머 ###########
def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  # 인코더에서 패딩을 위한 마스크
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더에서 미래의 토큰을 마스크 하기 위해서 사용합니다.
  # 내부적으로 패딩 마스크도 포함되어져 있습니다.
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 두 번째 어텐션 블록에서 인코더의 벡터들을 마스킹
  # 디코더에서 패딩을 위한 마스크
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # 인코더
  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  # 디코더
  global dec_outputs
  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # 완전연결층
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

########### 손실 함수 ###########
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

########### 커스텀된 학습률 ###########
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
sample_learning_rate = CustomSchedule(d_model=128)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
  return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

# model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# def decoder_inference(sentence):
# #   sentence = preprocess_sentence(sentence)

#   # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
#   # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
#   sentence = tf.expand_dims(
#       START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

#   # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
#   # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
#   output_sequence = tf.expand_dims(START_TOKEN, 0)

#   # 디코더의 인퍼런스 단계
#   for i in range(MAX_LENGTH):
#     # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
#     predictions = model(inputs=[sentence, output_sequence], training=False)
#     predictions = predictions[:, -1:, :]

#     # 현재 예측한 단어의 정수m
#     predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

#     # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
#     if tf.equal(predicted_id, END_TOKEN[0]):
#       break

#     # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
#     # 이 output_sequence는 다시 디코더의 입력이 됩니다.
#     output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

#   return tf.squeeze(output_sequence, axis=0)

# def sentence_generation(sentence):
#   # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
#   prediction = decoder_inference(sentence)

#   # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
#   predicted_sentence = tokenizer.decode(
#       [i for i in prediction if i < tokenizer.vocab_size])

#   print('입력 : {}'.format(sentence))
#   print('출력 : {}'.format(predicted_sentence))

#   return predicted_sentence


##########################################################################################################
def home(request):
    context = {}
    return render(request, "chathome.html", context)

@csrf_exempt
def chattrain(request):
    context = {}

    print('chattrain ---> +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    tf.keras.backend.clear_session()

    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model.summary()

    context['result_msg'] = 'Success'
    return JsonResponse(context, content_type="application/json")

model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
model.load_weights('static/chat_model/new8000_model.h5')

def decoder_inference(sentence):
#   sentence = preprocess_sentence(sentence)

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
  output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
  for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
    predictions = model(inputs=[sentence, output_sequence], training=False)
    predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수m
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
    output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

  return tf.squeeze(output_sequence, axis=0)

# def navi_flow():
#   msg =  "고장이면 1번, 분실이면 2번"
#   return msg

def get_category(sub_ty):
    ty_dict = {}
    error_msg = "추출 된 문제: " + sub_ty + "<br> 현재 등록되지 않은 시나리오 입니다."

    if sub_ty == '내비게이션':
      ty_dict['msg'] = '"' + sub_ty + '"' + " 관련 불편 사항을 겪고 계시군요, 죄송합니다.<br> 다음 중 어떤 상황 이신가요? 아래 보기 중 클릭해주세요."
      ty_dict['bool'] = True
      return ty_dict
    else:
      ty_dict['msg'] = error_msg
      ty_dict['bool'] = False
      return ty_dict

# def for_broken(clicked_broken):
#   cb_dict = {}
#   if clicked_broken == "고장":
#     cb_dict['msg'] = '내비게이션이 고장이 나서 매우 불편하셨겠습니다.. <br> 혹시 내비게이션을 껐다 켜 보셔도 동일하게 안되나요?'
#     cb_dict['bool'] = True
#     return cb_dict
#   else:
#     cb_dict['msg'] = '내비게이션을 분실하셔서 매우 당황하셨겠습니다.. <br> 혹시 괜찮으시다면 개인 휴대폰으로 사용 가능하실까요?'
#     cb_dict['bool'] = False
#     return cb_dict

@csrf_exempt
def chatanswer(request):
    context = {}
    questext = request.GET['questext']
    # questext = request
    print("questtext:", questext)
    # def sentence_generation(questext):
    # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.

    prediction = decoder_inference(questext)

    # # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    # print('입력 : {}'.format(sentence))
    # print('출력 : {}'.format(predicted_sentence))

    # return predicted_sentence

    # anstext = predicted_sentence
    # print(anstext)
    my_msg = get_category(predicted_sentence)['msg']
    my_bool = get_category(predicted_sentence)['bool']

    context['anstext'] = my_msg
    context['flag'] = '0'
    context['isNavy'] = my_bool

    return JsonResponse(context, content_type="application/json")
    # return anstext

# def sentence_generation(questext):
#     #입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
#     prediction = decoder_inference(questext)

#     #정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
#     predicted_sentence = tokenizer.decode(
#         [i for i in prediction if i < tokenizer.vocab_size])

#     print('입력 : {}'.format(sentence))
#     print('출력 : {}'.format(predicted_sentence))

#     return predicted_sentence

# print(sentence_generation('내비게이션이 안돼요'))
# print(chatanswer('내비게이션이 안돼요'))
# sub_type_mapping = {
#     '내비게이션': 0,
#     '타이어': 1,
#     '라이트': 2,
#     '시동': 3,
#     '경고등': 4,
#     '차량외부': 5,
#     '차량내부': 6,
#     '주행관련': 7,
#     '사고조사': 8,
#     '단말기': 9,
#     '주유/충전카드': 10,
#     '후방카메라': 11,
#     '하이패스': 12,
#     '차량점검': 13,
#     '브레이크': 14,
#     '블랙박스': 15,
#     '위생문제': 16,
#     '주차장': 17,
#     'ADAS': 18,
#     '비치품': 19,
#     '충전기확인': 20
# }
