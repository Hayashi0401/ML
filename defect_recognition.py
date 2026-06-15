
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

IMAGE_SIZE = 200
CLASS_NAMES = ['good', 'oil', 'scratch', 'stain']
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
TEST_RATIO = 0.2
RANDOM_SEED = 42


def find_dataset_dir():
    dataset_candidates = [
        os.path.join(SCRIPT_DIR, 'dataset'),
        os.path.join(PROJECT_DIR, 'dataset'),
    ]

    for dataset_dir in dataset_candidates:
        if os.path.isdir(dataset_dir):
            return dataset_dir

    return dataset_candidates[0]


DATASET_DIR = find_dataset_dir()
CLASS_DIRS = [os.path.join(DATASET_DIR, class_name) for class_name in CLASS_NAMES]
JUDGE_DIR = os.path.join(SCRIPT_DIR, 'JUDGE')


def load_images(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f'フォルダが見つかりません: {folder_path}\n'
            'Kaggleデータセットを解凍し、dataset/good, dataset/oil, '
            'dataset/scratch, dataset/stain の構成にしてください。'
        )

    images = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f'画像を読み込めませんでした: {image_path}')
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)

    return images


def split_images(images, test_ratio):
    if len(images) < 2:
        raise ValueError('学習とテストに分けるには、各クラス2枚以上の画像が必要です。')

    split_index = max(1, int(len(images) * (1 - test_ratio)))
    split_index = min(split_index, len(images) - 1)
    return images[:split_index], images[split_index:]


def make_dataset():
    np.random.seed(RANDOM_SEED)

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for label, class_dir in enumerate(CLASS_DIRS):
        images = load_images(class_dir)
        images = list(np.random.permutation(images))
        class_train, class_test = split_images(images, TEST_RATIO)

        train_images.extend(class_train)
        train_labels.extend([label] * len(class_train))
        test_images.extend(class_test)
        test_labels.extend([label] * len(class_test))

    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = make_dataset()

class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
total_train = len(y_train)
class_weight = {
    label: total_train / (len(CLASS_NAMES) * count)
    for label, count in enumerate(class_counts)
}

rand_index = np.random.permutation(np.arange(len(X_train)))
X_train = X_train[rand_index]
y_train = y_train[rand_index]

img_JUDGE = []
if os.path.isdir(JUDGE_DIR):
    img_JUDGE = load_images(JUDGE_DIR)

# categorical_crossentropyとともに用いるためのバイナリのクラス行列に変換
y_train = to_categorical(y_train, num_classes=len(CLASS_NAMES))
y_test = to_categorical(y_test, num_classes=len(CLASS_NAMES))



# モデルにVGG16を使用
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# vggのoutputを受け取り、4クラス分類する層を定義します
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(len(CLASS_NAMES), activation='softmax'))

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
model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=3,
    validation_data=(X_test, y_test),
    class_weight=class_weight
)

# 重みを保存
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
model.save(os.path.join(result_dir, 'model.h5'))




# 画像を一枚受け取り、欠陥種別を判定する関数

def pred_quality(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    pred = np.argmax(model.predict(np.array([img])))
    return CLASS_NAMES[pred]
    

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



print(f'使用データセット: {DATASET_DIR}')
print(f'学習データ: {len(X_train)}枚')
print(f'テストデータ: {len(X_test)}枚')
print(f'クラス: {CLASS_NAMES}')

# 任意の検証用データでテストを行う
for i in img_JUDGE[:15]:
    result = pred_quality(i)
    print(f"判定結果: {result}")
