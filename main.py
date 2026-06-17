import os
import base64
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np

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

model = load_model('./results/model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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
                

            return render_template("index.html", answer=pred_answer, img_base64=img_base64)

    return render_template("index.html", answer="", img_base64="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
