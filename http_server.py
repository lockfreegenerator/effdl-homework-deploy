from flask import Flask, request, jsonify
import requests
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
import io
import os
import json

app = Flask(__name__)

# Load the pretrained model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75)
model.eval()

#model = torch.jit.load('fasterRCNN.pt')
#model = torch.jit.load(f'{os.path.dirname(os.path.abspath(__file__))}/fasterRCNN.pt')
#model.eval()

# Get the COCO class names
categories = weights.meta["categories"]

# with open(f'{os.path.dirname(os.path.abspath(__file__))}/labels.json', 'r') as f:
#     labels_raw = json.loads(f.read())
#     categories = {int(index): value for index, value in enumerate(labels_raw)}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data['url']
    
    # Download the image
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    
    image_tensor = F.to_tensor(image)
    
    #print(image_tensor.shape)
    #print(categories)

    # Perform inference
    with torch.no_grad():
        predictions = model([image_tensor])
    
    print(predictions)

    labels = predictions[1][0]['labels'].tolist()
    objects = [categories[label] for label in labels]
    
    return jsonify({"objects": objects})


from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Define a counter for the number of inferences
inference_count = Counter('app_http_inference_count_total', 'Total number of HTTP endpoint invocations')

@app.route('/metrics')
def metrics():
    inference_count.inc()
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
