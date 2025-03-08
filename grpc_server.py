import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
import requests
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
import io


class InferenceServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.75)
        self.model.eval()
        self.categories = weights.meta["categories"]


    def Predict(self, request, context):
        image_url = request.url
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        labels = predictions[0]['labels'].tolist()
        objects = [self.categories[label] for label in labels]
        
        return inference_pb2.InstanceDetectorOutput(objects=objects)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()



if __name__ == '__main__':
    serve()
