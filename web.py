from flask import Flask, request, jsonify
from flask_cors import CORS
from doclayout_yolo import YOLOv10
from PIL import Image
import io
import base64

# 1. 啟動時載入模型
model = YOLOv10("best.pt")  # 換成你的權重路徑

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def run_inference(img: Image.Image):
    """共用的推論函式，輸入 PIL Image，輸出 detections list。"""
    # results = model(img)[0]
    results = model.predict(
        img,   # Image to predict
        imgsz=1024,        # Prediction image size
        conf=0.2,          # Confidence threshold
        device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
            "class_name": cls_name,
        })
    return detections

def load_image_from_request():
    """
    優先使用 multipart/form-data 的 file，
    否則嘗試從 JSON 的 image_base64 讀取圖片。
    回傳 PIL Image，或丟出例外。
    """
    # 1. 檔案上傳模式 (multipart/form-data)
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            raise ValueError("No selected file")
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img

    # 2. JSON base64 模式 (application/json)
    if request.content_type and "application/json" in request.content_type:
        data = request.get_json(silent=True)
        if not data or "image_base64" not in data:
            raise ValueError("Missing 'image_base64' field in JSON body")

        b64_str = data["image_base64"]

        # 如果前端傳 dataURL 格式，例如：data:image/jpeg;base64,XXXX
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception:
            raise ValueError("Invalid base64 string")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img

    # 3. 兩種都沒有
    raise ValueError("No image found in request (file or image_base64)")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = load_image_from_request()
        detections = run_inference(img)

        return jsonify({
            "num_detections": len(detections),
            "detections": detections
        })

    except ValueError as ve:
        # 使用者輸入錯誤（400）
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # 其他後端錯誤（500）
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)
