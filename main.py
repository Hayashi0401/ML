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
ALL_CLASSES = classes + ["unknown"]
image_size = 200

UPLOAD_FOLDER = "uploads"
for cls in ALL_CLASSES:
    os.makedirs(
        os.path.join(UPLOAD_FOLDER, cls),
        exist_ok=True
    )

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
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
    original_img = img.copy()
    img = cv2.resize(img, (image_size, image_size))
    img_array = np.array([img.astype('float32')])
    
    # 最後の畳み込み層を取得
    last_conv_layer_name = 'block5_conv3'
    
    # Grad-CAMを計算するモデルを構築
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # 勾配を計算
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, class_index]
    
    # 最後の畳み込み層に対する勾配を計算
    grads = tape.gradient(class_channel, conv_outputs)
    
    # 勾配の平均をとって、重みとする
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Grad-CAMを計算
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ヒートマップを正規化（0-255）
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = tf.cast(heatmap * 255, tf.uint8).numpy()
    
    # ヒートマップを元のサイズにリサイズ
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    
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

                # ---------- フィルター ----------

                if confidence < 80:
                    pred_answer = "Unknown"

                elif pred_class == "good" and confidence < 98:
                    pred_answer = "Unknown"

                else:
                    pred_answer = pred_class

                # ---------- 保存 ----------

                save_dir = os.path.join(
                    UPLOAD_FOLDER,
                    pred_answer
                    if pred_answer in classes
                    else "unknown"
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
                    "class": pred_answer,
                    "confidence": round(confidence, 1)
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
    """Grad-CAMを計算して返すエンドポイント"""
    global last_uploaded_image, last_predicted_class
    
    if last_uploaded_image is None or last_predicted_class is None:
        return jsonify({'error': 'No image to analyze'}), 400
    
    try:
        gradcam_img = generate_gradcam(
            last_uploaded_image,
            model,
            last_predicted_class,
            image_size
        )
        return jsonify({'gradcam_image': gradcam_img})
    except Exception as e:
        print(f"Grad-CAM計算エラー: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
