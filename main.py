import os
import base64
from flask import Flask, request, redirect, render_template, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np
import cv2
import io
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))


classes = ["good", "oil", "scratch", "stain"]
ALL_CLASSES = classes 
image_size = 200

UPLOAD_FOLDER = "uploads"
# ensure upload folder and class subfolders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
for cls in ALL_CLASSES:
    os.makedirs(os.path.join(UPLOAD_FOLDER, cls), exist_ok=True)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_gradcam(img_path, model, class_index, image_size=200):
    """
    Grad-CAMを生成する関数
    
    Args:
        img_path: 画像ファイルのパス
        model: 学習済みモデル
        class_index: 対象クラスのインデックス
        image_size: 入力画像サイズ
    
    Returns:
        Base64エンコードされたGrad-CAM画像
    """
    # 画像の読み込みと前処理
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {img_path}")

    original_height, original_width = img.shape[:2]
    img_for_model = cv2.resize(img, (image_size, image_size))
    img_array = np.expand_dims(img_for_model.astype('float32'), axis=0)
    img_tensor = tf.cast(img_array, tf.float32)

    # 最後の畳み込み層名（モデルに依存するため存在確認を行う）
    last_conv_layer_name = 'block5_conv3'
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception:
        # フォールバック: モデル内の最後のConv層を探索
        conv_layers = [l for l in model.layers if 'conv' in l.name]
        if not conv_layers:
            raise RuntimeError('モデルに畳み込み層が見つかりません')
        last_conv_layer = conv_layers[-1]
        last_conv_layer_name = last_conv_layer.name

    # Grad-CAMを計算するモデルを構築
    grad_model = Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # 勾配を計算
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        # バッチ次元を残して該当クラスのスコアを取得
        class_channel = predictions[:, class_index]

    # 最後の畳み込み層に対する勾配を計算
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError('勾配が計算できませんでした')

    # 勾配の空間平均をとって、重みとする
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Grad-CAMを計算（チャンネルごとの重み付き和）
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # ヒートマップを正規化（0-255） — 0割り防止
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / (max_val + tf.keras.backend.epsilon())

    heatmap = tf.cast(heatmap * 255, tf.uint8).numpy()

    # ヒートマップを元の画像サイズにリサイズ
    heatmap = cv2.resize(heatmap, (original_width, original_height))

    # カラーマップを適用
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 元の画像とヒートマップをブレンド
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    # PIL画像に変換
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(superimposed_img_rgb)

    # Base64エンコード
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"


model = load_model('./results/model.h5')

# 最後にアップロードされた画像の情報を保持
last_uploaded_image = None
last_predicted_class = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global last_uploaded_image, last_predicted_class

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)

        files = request.files.getlist('file')

        if len(files) == 0:
            flash('ファイルがありません')
            return redirect(request.url)

        results = []

        for file in files:

            if file.filename == '':
                continue

            if file and allowed_file(file.filename):

                filename = secure_filename(file.filename)

                # AI用画像
                file_bytes = io.BytesIO(file.read())
                img = image.load_img(
                    file_bytes,
                    target_size=(image_size, image_size)
                )

                img = image.img_to_array(img)
                data = np.array([img])

                result = model.predict(data, verbose=0)[0]

                predicted = result.argmax()
                confidence = result[predicted] * 100

                pred_class = classes[predicted]

                # ---------- 保存 ----------

                save_dir = os.path.join(
                    UPLOAD_FOLDER,
                    pred_class
                )

                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(
                    save_dir,
                    filename
                )

                file.seek(0)
                file.save(save_path)

                # Grad-CAM用（最後の1枚）
                last_uploaded_image = save_path
                last_predicted_class = predicted

                results.append({
                    "filename": filename,
                    "class": pred_class,
                    "confidence": float(round(confidence, 1))
                })

        return render_template(
            "index.html",
            results=results
        )

    return render_template(
        "index.html",
        results=[]
    )


@app.route('/gradcam', methods=['POST'])
def get_gradcam():
    """Grad-CAMを計算して返すエンドポイント（最新画像、または過去画像に対応）"""
    global last_uploaded_image, last_predicted_class
    
    # JSONリクエスト（過去画像用）か、通常のPOST（直近画像用）かを判定
    data = request.get_json() if request.is_json else {}
    filename = data.get('filename')
    class_name = data.get('class_name')
    
    if filename and class_name:
        # 過去履歴から指定された画像を呼び出す場合
        class_name_lower = class_name.lower()
        img_path = os.path.join(UPLOAD_FOLDER, class_name_lower, filename)
        
        if class_name_lower in classes:
            class_index = classes.index(class_name_lower)
        else:
            return jsonify({'error': '対象外(unknown)のGrad-CAMは生成できません'}), 400
    else:
        # 直近で判定した画像の場合（フォールバック）
        img_path = last_uploaded_image
        class_index = last_predicted_class

    if img_path is None or class_index is None or not os.path.exists(img_path):
        return jsonify({'error': '画像が見つかりません。再度判定を行ってください。'}), 400
    
    try:
        gradcam_img = generate_gradcam(
            img_path,
            model,
            class_index,
            image_size
        )
        return jsonify({'gradcam_image': gradcam_img})
    except Exception as e:
        print(f"Grad-CAM生成エラー: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history/<class_name>', methods=['GET'])
def get_history(class_name):
    """指定されたクラスのアップロード履歴（ファイル名リスト）を新しい順に返す"""
    class_name_lower = class_name.lower()
    if class_name_lower not in [cls.lower() for cls in ALL_CLASSES]:
        return jsonify({'error': '無効なクラス名です'}), 400
        
    target_dir = os.path.join(UPLOAD_FOLDER, class_name_lower)
    if not os.path.exists(target_dir):
        return jsonify({'files': []})
        
    try:
        files = [f for f in os.listdir(target_dir) if allowed_file(f)]
        # 更新日時（アップロード日時）の新しい順にソート
        files.sort(key=lambda x: os.path.getmtime(os.path.join(target_dir, x)), reverse=True)
        return jsonify({'files': files})
    except Exception as e:
        print(f"履歴取得エラー: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
