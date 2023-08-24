import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv 파일 열기
ttt_data_csv = pd.read_csv("./tic-tac-toe.csv")

# o, b, x를 각각 1, 0, -1로 대체 / True, False를 각각 1, 0으로 교체
ttt_data_csv.replace(to_replace = "o", value = "1", inplace=True)
ttt_data_csv.replace(to_replace = "b", value = "0", inplace=True)
ttt_data_csv.replace(to_replace = "x", value = "-1", inplace=True)

# index 속성에 False 값을 주고 "tic-tac-toe-conv.csv" 파일에 저장
ttt_data_csv.to_csv("tic-tac-toe-conv.csv", index = False)

# 수정된 csv 파일 열기
def load_ttt(shuffle = False):
    # 항목 인덱스 9의 문자열을 {'True' : 1, 'False' : 0}의 정수 레이블로 변환
    label = {'True' : 1, 'False' : 0}
    data = np.loadtxt("./tic-tac-toe-conv.csv", skiprows = 1, delimiter = ",",
                      converters = {9: lambda name: label[name.decode()]})
    # shuffle = True이면 순서를 섞은 후 반환
    if shuffle:
        np.random.shuffle(data)
    return data

# 훈련 데이터 70%(670개), 테스트 데이터 30%(288개)로 분리 
def train_test_data_set(ttt_data, test_rate = 0.3):
    n = int(ttt_data.shape[0] * (1 - test_rate))
    x_train = ttt_data[:n, :-1]
    y_train = ttt_data[:n, -1]

    x_test = ttt_data[n:, :-1]
    y_test = ttt_data[n:, -1]

    return (x_train, y_train), (x_test, y_test)

ttt_data = load_ttt(shuffle = True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(ttt_data, test_rate = 0.3)

# 각 데이터 개수 출력 
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:",  x_test.shape)
print("y_test.shape:",  y_test.shape)

# 손실 함수가 MSE 또는 categorical_crossentropy 이라면
# tf.keras.utils.to_categorical()을 통해 y_train과 y_test를 One-Hot 인코딩으로 변환
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 은닉층의 뉴런 개수 
n = 10
model = tf.keras.Sequential()
# 입력 데이터 9개
model.add(tf.keras.layers.Dense(units=n, input_dim=9, activation='sigmoid'))
# 출력층의 활성 함수가 activation='softmax'인 신경망 생성 / 출력 데이터 2
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary()

# MSE 함수 정의
def MSE(y, t):
    return tf.reduce_mean(tf.square(y - t))

# categorical_crossentropy
CCE = tf.keras.losses.CategoricalCrossentropy()

# learning_rate 0.01로 설정 / optimizer RMSprop로 설정  
opt = RMSprop(learning_rate=0.01)
# opt = Adam(learning_rate=0.01)
# opt = SGD(learning_rate=0.01)
# opt = Adagrad(learning_rate=0.01)

# loss
# model.compile(optimizer=opt, loss= MSE, metrics=['accuracy'])
model.compile(optimizer=opt, loss= CCE, metrics=['accuracy'])

# epochs 100으로 설정
ret = model.fit(x_train, y_train, epochs=100, verbose=0)

print("len(model.layers):", len(model.layers))
loss = ret.history['loss']
accuracy = ret.history['accuracy']

# loss 변화율 출력 
plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# accyracy 변화율 출력 
plt.plot(accuracy)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis = 1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis = 1), y_label)
print("confusion_matrix(C):", C)
