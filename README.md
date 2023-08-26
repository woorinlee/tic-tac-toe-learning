# tic-tac-toe-learning(삼목게임 학습)

## 개요

[datasets/tic-tac-toe](https://github.com/datasets/tic-tac-toe)

tic-tac-toe.csv의 삼목 게임을 tensorflow로 학습한다.

전체 959개의 데이터 중 70%를 학습 데이터로, 30%를 테스트 데이터로 사용하며 learning_rate, epoch, 최적화 알고리즘, 손실 함수에 따라 달라지는 결과를 비교하도록 한다.

손실 함수로는 평균 제곱 
오차(MSE), 다중 분류(CCE; categorical_crossentropy)를 사용하며, 최적화 알고리즘으로는 RMSprop(), Adam(), SDG(), Adagrad()를 사용한다. 

## 준비 과정

```
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

학습 전, 필요한 패키지들을 import한다.

```
ttt_data_csv = pd.read_csv("./tic-tac-toe.csv")

ttt_data_csv.replace(to_replace = "o", value = "1", inplace=True)
ttt_data_csv.replace(to_replace = "b", value = "0", inplace=True)
ttt_data_csv.replace(to_replace = "x", value = "-1", inplace=True)

ttt_data_csv.to_csv("tic-tac-toe-conv.csv", index = False)
```

|기존 데이터|대체 데이터|
|:---:|:---:|
|o|1|
|b|0|
|x|-1|

pandas를 통해 tic-tac-toe.csv의 o, b, x를 각각 1, 0, -1로 대체한 후 저장한다.

```
def load_ttt(shuffle = False):
    label = {'True' : 1, 'False' : 0}
    data = np.loadtxt("./tic-tac-toe-conv.csv", skiprows = 1, delimiter = ",",
        converters = {9: lambda name: label[name.decode()]})
    if shuffle:
        np.random.shuffle(data)
    return data
```

numpy의 loadtxt 함수를 통해 csv 파일을 열고, 항목 인덱스 9의 문자열을 {'True' : 1, 'False' : 0}의 정수 레이블로 변환한다. 이후 numpy의 random.shuffle을 통해 매개변수로 True가 입력되면 변환된 내용의 순서를 섞은 후 반환한다.

```
def train_test_data_set(ttt_data, test_rate = 0.3):
    n = int(ttt_data.shape[0] * (1 - test_rate))
    x_train = ttt_data[:n, :-1]
    y_train = ttt_data[:n, -1]

    x_test = ttt_data[n:, :-1]
    y_test = ttt_data[n:, -1]

    return (x_train, y_train), (x_test, y_test)
```

전체 959개의 데이터 중 70%를 학습 데이터로, 30%를 테스트 데이터로 사용하도록 한다.

```
def MSE(y, t):
    return tf.reduce_mean(tf.square(y - t))

CCE = tf.keras.losses.CategoricalCrossentropy()
```

손실 함수 MSE, CCE를 작성한다.

```
ttt_data = load_ttt(shuffle = True)
(x_train, y_train), (x_test, y_test) = train_test_data_set(ttt_data, test_rate = 0.3)

print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:",  x_test.shape)
print("y_test.shape:",  y_test.shape)
```

load_ttt 함수에 shuffle 값으로 True를 입력한 후 반환된 값을 ttt_data에 저장한다. 이후 해당 값을 train_test_data_set 함수를 통해 학습 데이터와 테스트 데이터로 변환한다. 이후 각 데이터의 개수를 출력한다.

```
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
```

만약 손실 함수가 MSE 또는 CCE 이라면 tf.keras.utils.to_categorical()을 통해 y_train과 y_test를 One-Hot 인코딩으로 변환하도록 한다.

```
n = 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=n, input_dim=9, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary()
```

은닉층의 뉴런 개수를 10개로 하고, 입력 데이터를 9개, 출력 데이터를 2개로 하며 출력층의 활성 함수가 activation=‘softmax’인 신경망을 생성한다.

## 학습 과정

```
opt = RMSprop(learning_rate=0.01)
# opt = Adam(learning_rate=0.01)
# opt = SGD(learning_rate=0.01)
# opt = Adagrad(learning_rate=0.01)
```

최적화 알고리즘을 설정한다. (RMSprop)


```
model.compile(optimizer=opt, loss= MSE, metrics=['accuracy'])
# model.compile(optimizer=opt, loss= CCE, metrics=['accuracy'])
```

손실 함수를 설정한다. (MSE)

```
ret = model.fit(x_train, y_train, epochs=100, verbose=0)

print("len(model.layers):", len(model.layers))
loss = ret.history['loss']
accuracy = ret.history['accuracy']
```

epochs 값을 100으로 설정한다.

```
plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

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
```

loss 변화율과 accuracy 변화율을 출력하고 손실율, 정확도를 출력한다.

## 학습 결과

RMSprop, learning_late = 0.01, MSE, epochs = 100에 대한 결과는 다음과 같다.

|loss 변화율|accuracy 변화율|
|:---:|:---:|
|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e3102a01-b103-447e-9b38-890f6ea2fd36"/>|<img width="95%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/291a1929-14f7-4d09-9039-54f2531e643b"/>|

|출력 내용|
|:---:|
|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/85390bbb-74c5-497b-ae2a-5a2a952e587f"/>|

|훈련 데이터 정확도|테스트 데이터 정확도|
|---:|---:|
|100%|99.65%|

## 학습 결과 비교 분석

### 1. learning_late

기존 learning_late 값인 0.01보다 큰 0.1 값을 사용하여 결과를 도출한다.

||loss 변화율|accuracy 변화율|
|:---|:---:|:---:|
|learning_late=0.01|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e3102a01-b103-447e-9b38-890f6ea2fd36"/>|<img width="95%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/291a1929-14f7-4d09-9039-54f2531e643b"/>|
|learning_late=0.1|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/a7aa2e55-ddb5-4ad1-863d-a38a3167f475"/>|<img width="93%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/ece3a16b-6a29-4ea1-bca5-63adf337fe86"/>|

||출력 내용|
|:---|:---:|
|learning_late=0.01|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/85390bbb-74c5-497b-ae2a-5a2a952e587f"/>|
|learning_late=0.1|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/86b7e559-993b-4bcd-b1ba-b34533eb0bb8"/>|

||훈련 데이터 정확도|테스트 데이터 정확도|
|:---|---:|---:|
|learning_late=0.01|100%|99.65%|
|learning_late=0.1|100%|98.26%|

### 2. epochs

기존 epochs 값인 100보다 작은 200 값을 사용하여 결과를 도출한다.

||loss 변화율|accuracy 변화율|
|:---|:---:|:---:|
|epochs=100|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e3102a01-b103-447e-9b38-890f6ea2fd36"/>|<img width="95%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/291a1929-14f7-4d09-9039-54f2531e643b"/>|
|epochs=200|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/20053648-1a87-4963-b936-d074c59492c6"/>|<img width="93%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/d62ea108-6393-4a7e-a93c-33db7203d244"/>|

||출력 내용|
|:---:|:---:|
|epochs=100|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/85390bbb-74c5-497b-ae2a-5a2a952e587f"/>|
|epochs=200|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/8b8cb70e-42fc-4855-ab56-9216094f5f3c"/>|

||훈련 데이터 정확도|테스트 데이터 정확도|
|:---|---:|---:|
|epochs=100|100%|99.65%|
|epochs=200|100%|97.57%|

### 3. 최적화 알고리즘

RMSprop(), Adam(), SDG(), Adagrad() 최적화 알고리즘에 대한 결과를 도출한다.

||loss 변화율|accuracy 변화율|
|:---:|:---:|:---:|
|RMSprop()|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e3102a01-b103-447e-9b38-890f6ea2fd36"/>|<img width="95%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/291a1929-14f7-4d09-9039-54f2531e643b"/>|
|Adam()|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/7df2f256-e891-4f0c-a6de-33245798d2ce"/>|<img width="94%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/c1bd0ed5-15b6-4af6-9151-0fece7e4d939"/>|
|SDG()|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/f3ea2a2b-cb24-478e-8a0c-0d111a0452b1"/>|<img width="93%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/96bf3630-0fa1-4cac-979b-b33eef3610e0"/>|
|Adagrad()|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/8996f0fe-e909-46ed-9805-6808948d5318"/>|<img width="94%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/59b90422-bb71-4424-915f-bd3c6df57932"/>|

||출력 내용|
|:---:|:---:|
|RMSprop()|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/85390bbb-74c5-497b-ae2a-5a2a952e587f"/>|
|Adam()|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e49c3114-51db-49a1-886a-ff78c62050cf"/>|
|SDG()|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/6f7abe96-2b1e-4643-9adf-2df74e9c7142"/>|
|Adagrad()|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/9c994f4a-b317-4d9a-945e-ce463178fb74"/>|

||훈련 데이터 정확도|테스트 데이터 정확도|
|:---|---:|---:|
|RMSprop()|100%|99.65%|
|Adam()|99.85%|98.26%|
|SDG()|70%|67.01%|
|Adagrad()|75.07%|76.39%|

### 4. 손실 함수

MSE, CCE 손실 함수에 대한 결과를 도출한다.

||loss 변화율|accuracy 변화율|
|:---:|:---:|:---:|
|MSE|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/e3102a01-b103-447e-9b38-890f6ea2fd36"/>|<img width="95%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/291a1929-14f7-4d09-9039-54f2531e643b"/>|
|CCE|<img width="98%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/f83f2b00-f86e-44bb-ac8f-a63807cdd6d3"/>|<img width="94%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/8ea9820f-e399-4a36-951f-0962f5f37692"/>|

||출력 내용|
|:---:|:---:|
|MSE|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/85390bbb-74c5-497b-ae2a-5a2a952e587f"/>|
|CCE|<img width="80%" src="https://github.com/woorinlee/tic-tac-toe-learning/assets/83910204/926d2d1d-c8ff-407c-a045-c72886d7bb8d"/>|

||훈련 데이터 정확도|테스트 데이터 정확도|
|:---|---:|---:|
|MSE|100%|99.65%|
|CCE|100%|98.61%|