import os, sys
import torch, json
import numpy as np
import argparse
import matplotlib.pyplot as plt
# Aramak istediğiniz klasör adı
"""target_folder_name = "DINO"

# Şu anki çalışma dizini üzerinde arama
for root, dirs, files in os.walk(os.getcwd()):
    if target_folder_name in dirs:
        dino_folder_path = os.path.join(root, target_folder_name)
        print("DINO Klasörü Yolu:", dino_folder_path)
        break

sys.path.append(dino_folder_path)"""

dino_folder_path = os.path.join(os.getcwd(), "DINO")

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

from PIL import Image
import os
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(800, scale=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(model_path, model_config_path=os.path.join(os.getcwd(), "DINO", "config", "DINO", "DINO_4scale.py")):
  
    #model_config_path = os.path.join(root, "DINO/config/DINO/DINO_4scale.py")
    model_checkpoint_path = model_path

    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    return model, postprocessors

def preprocess_image(image_path):
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    return image

def main():
    
    parser = argparse.ArgumentParser(description='seq2seq model')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--image', type=str, required=True, help='image path')
   
    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    config = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, postprocessors = load_model(model_path, config)

    image = preprocess_image(image_path)

    output = model.cuda()(image[None].cuda())

    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    
    thershold = 0.05

    vslzr = COCOVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    
    select_mask = scores > thershold
    print(boxes[select_mask])
    box_label = ["fire" for i in range(len(boxes[select_mask]))]
    pred_dict = {
        'image_id': 4,
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label
    }

    vslzr.visualize(image, pred_dict, savedir="", dpi=200)

if __name__ == "__main__":
    main()