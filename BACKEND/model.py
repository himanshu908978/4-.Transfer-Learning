import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from pathlib import Path
import os
from PIL import Image

BaseDir = Path(__file__).resolve().parent.parent

VGG_weights = BaseDir/"MODELS"/"VGG_0_0.08.pth"
Resnet_weights = BaseDir/"MODELS"/"ResNet_0_0.08.pth"
EfficientNet_weights = BaseDir/"MODELS"/"EfficientNet_1_0.06.pth"




model = models.vgg16(weights = None)
for params in model.features.parameters():
  params.requires_grad = False
model.classifier[6] = nn.Linear(4096,2)

model1 = models.resnet18(weights = None)
for param in model1.parameters():
  param.requires_grad = False
model1.fc = nn.Linear(model1.fc.in_features,2)

model2 = models.efficientnet_b3(weights = None)
for param in model2.parameters():
  param.requires_grad = False
model2.classifier[1] = nn.Linear(model2.classifier[1].in_features,2)

model.load_state_dict(torch.load(VGG_weights,map_location="cpu"))
model.eval()
model1.load_state_dict(torch.load(Resnet_weights,map_location="cpu"))
model1.eval()
model2.load_state_dict(torch.load(EfficientNet_weights,map_location="cpu"))
model2.eval()

model_list = [model,model1,model2]


def inference(model,inputimg):
  inputimg = Image.open(inputimg).convert("RGB")
  transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])
  inputimg = transform(inputimg)
  inputimg = inputimg.unsqueeze(0)
  output = model(inputimg)

  probabilities = torch.softmax(output,dim=1)
  pred_class = torch.argmax(probabilities,dim=1)
  conf = probabilities[0][pred_class]
  return pred_class.item(),conf.item()