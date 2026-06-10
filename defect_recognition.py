
# ライブラリのインポート
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
import glob

# ディレクトリの変更
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ファイルパスの設定
path_OK = os.listdir('./ok_front')
path_DEFECT = os.listdir('./def_front')
path_JUDGE = os.listdir('./JUDGE')

# データを合格、不合格、検証用に分類
img_OK = []
img_DEFECT = []
img_JUDGE=[]

for i in range(len(path_OK)):
    img = cv2.imread('./ok_front/' + path_OK[i])
    img = cv2.resize(img, (200,200))
    img_OK.append(img)

for i in range(len(path_DEFECT)):
    img = cv2.imread('./def_front/' + path_DEFECT[i])
    img = cv2.resize(img, (200,200))
    img_DEFECT.append(img)

# 検証用データ
for i in range(len(path_JUDGE)):
    img = cv2.imread('./JUDGE/' + path_JUDGE[i])
    img = cv2.resize(img, (200,200))
    img_JUDGE.append(img)

X = np.array(img_OK + img_DEFECT)
y = np.array([0]*len(img_OK) + [1]*len(img_DEFECT))


# 画像をシャッフル
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]

# データの分割
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

# categorical_crossentropyとともに用いるためのバイナリのクラス行列に変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# ニューラスイン！！
# モデルにvggを使用
input_tensor = Input(shape=(200, 200, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# vggのoutputを受け取り、2クラス分類する層を定義します
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

# vggと、top_modelを連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vggの層の重みを変更不能にします
for layer in model.layers[:19]:
    layer.trainable = False

# コンパイルします
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              metrics=['accuracy'])



# 学習を行います
model.fit(X_train, y_train, batch_size=16, epochs=3, validation_data=(X_test, y_test))

# 重みを保存
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
model.save(os.path.join(result_dir, 'model.h5'))




# 画像を一枚受け取り、合格か不合格かを判定する関数

def pred_quality(img):
    img = cv2.resize(img, (200, 200))
    pred = np.argmax(model.predict(np.array([img])))
    if pred == 0:
        return 'OK'
    else:
        return 'DEFECT'
    

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



# 検証用データ10枚でテストを行う
# 検証用データでテストを行う
for i in img_JUDGE[:15]:
    # 自作した関数を使って、画像がOKかDEFECTかを判定
    result = pred_quality(i)
    
    # 判定結果をターミナル（コンソール）に表示
    print(f"判定結果: {result}")
    
    # ローカル環境用の画像表示処理
    # ウィンドウ名に判定結果を表示すると分かりやすいです
    cv2.imshow(f'Judge Image - Result: {result}', i)
    cv2.waitKey(0)  # キーボードのキーが押されるまで待機
    cv2.destroyAllWindows()  # ウィンドウを閉じる
