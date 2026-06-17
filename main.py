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
image_size = 200

UPLOAD_FOLDER = "uploads"
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
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            with open(filepath, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            ext = filename.rsplit('.', 1)[1].lower()
            mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            img_base64 = f"data:{mime_type};base64,{encoded_string}"

            img = image.load_img(filepath, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            
            result = model.predict(data)[0]
            predicted = result.argmax()
            confidence = result[predicted] * 100
            
            # --- ここから書き換え（判定フィルターの強化） ---
            
            # 1. そもそもAIの信頼度が低すぎる（80%未満）場合は一律ではじく
            if confidence < 80.0:
                pred_answer = "スマホではないか、または判別できない異物です"
                
            # 2. 「good（正常）」と判定されたが、信頼度が98%未満の場合
            # （顔写真など、AIがなんとなく綺麗だからと勘違いした画像はここで弾きます）
            elif classes[predicted] == "good" and confidence < 98.0:
                pred_answer = "スマホではないか、または判別できない異物です（確信度が足りません）"
                
            # 3. 上記の条件をクリアした、自信のある正規の判定結果
            else:
                pred_answer = f"これは {classes[predicted]} です（信頼度: {confidence:.1f}%）"
                

            # 後でGrad-CAM計算用に保存
            last_uploaded_image = filepath
            last_predicted_class = predicted

            return render_template("index.html", answer=pred_answer, img_base64=img_base64)

    return render_template("index.html", answer="", img_base64="")


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
