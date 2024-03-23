# 必要なモジュールを読み込む
# Flask関連
from flask import Flask, render_template, request, flash, session
import os

# PyTorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models

# Pillow(PIL)、datetime
from PIL import Image
from datetime import datetime

# GPUチェック
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデルの作成
model = models.resnet18(pretrained = True).to(device)
# 学習モデルをロードする
model.load_state_dict(torch.load('./net.prm'))

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = "./static/uploads"

app.secret_key = "mushroom"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    #if request.files["file"]:
    if request.method == "POST":
        if 'upload_file' not in request.files:
            flash('ファイルを選択してください')
        # アップロードされたファイルをいったん保存する
        f = request.files["upload_file"]
        if f.filename == '':
            flash('ファイルを選択してください')
        else:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
            f.save(filepath)
            print(filepath)
            # 画像ファイルを読み込む
            image = Image.open(filepath)
            transform = transforms.Compose(
                [
                    transforms.Resize(256),  # (256) で切り抜く。
                    transforms.ToTensor(),  # テンソルにする。
                ]
            )
            image = image.convert("RGB")
            input = transform(image).to(device)
            input = input.unsqueeze(0).to(device)
            # 予測を実施
            model.eval()
            output = model(input)
            prediction = torch.max(output, 1)
            result = prediction[1].item()

            # 結果に応じてページの遷移先を決定
            if result == 1:
                return render_template("./result.html", filepath=filepath)
            else:
                return render_template("./result2.html", filepath=filepath)
        
    return render_template("./index.html")

if __name__ == "__main__":
    app.run(debug=True)
