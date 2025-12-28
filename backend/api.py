from flask import Flask, request, jsonify
from PIL import Image
import io, base64, torch
import torch.nn.functional as F
from model_architecture import EnhancedCNN
import torchvision.transforms as transforms

app = Flask(__name__)

# CORS
@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Methods'] = '*'
    return r

print("جاري تحميل الموديل...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = r"C:\Users\ahdsy\Desktop\CV_Project\backend\models\best_model_enhanced.pth"

checkpoint = torch.load(weights_path, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

model = EnhancedCNN(num_classes=2, dropout_rate=0.4)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("تم تحميل الموديل بنجاح ✓")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(img):
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        prob = F.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)
    label = "with_mask" if idx.item() == 0 else "without_mask"
    return label, round(conf.item()*100, 1)

@app.route("/api/predict", methods=["POST"])
def upload():
    img = Image.open(request.files['file'].stream).convert("RGB")
    pred, conf = predict(img)
    return jsonify({"prediction": pred, "confidence": conf})

@app.route("/api/realtime", methods=["POST"])
def realtime():
    img_data = request.json['image'].split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("RGB")
    pred, conf = predict(img)
    return jsonify({
        "prediction": pred,
        "confidence": conf
    })

if __name__ == "__main__":
    app.run(port=8000, debug=False)