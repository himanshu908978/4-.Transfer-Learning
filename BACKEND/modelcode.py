# THIS CODE IS WRITTEN IN GOOGLE COLAB AND MODEL TRAINED IN THE GOOGLE COLAB NOTEBOOK AND DATASET IN IN GOOGLE DRIVE

'''
from torchvision import models
from torchvision import transforms
from torchvision.models import VGG16_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models import EfficientNet_B3_Weights
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])

dataset_path = "/content/drive/MyDrive/Animals"
dataset = ImageFolder(dataset_path,transform = transform)

print("The Length of Dataset is :- ",len(dataset))
train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

train_dataset,test_dataset = random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset,batch_size=32,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle = False)


model = models.vgg16(weights = VGG16_Weights.DEFAULT)
for params in model.features.parameters():
  params.requires_grad = False
model.classifier[6] = nn.Linear(4096,2)
model = model.to(device)
lr = 0.001
epochs = 3
optim = torch.optim.Adam(model.classifier.parameters(),lr = lr)
loss_fn = nn.CrossEntropyLoss()

model1 = models.resnet18(weights = ResNet18_Weights.DEFAULT)
for param in model1.parameters():
  param.requires_grad = False
model1 = model1.to(device)
model1.fc = nn.Linear(model1.fc.in_features,2)
optim1 = torch.optim.Adam(model1.fc.parameters(),lr = lr)

model2 = models.efficientnet_b3(weights = EfficientNet_B3_Weights.DEFAULT)
for param in model2.parameters():
  param.requires_grad = False
model2 = model2.to(device)
model2.classifier[1] = nn.Linear(model2.classifier[1].in_features,2)
optim2 = torch.optim.Adam(model2.classifier[1].parameters(),lr =lr)

model_list = [model,model1,model2]
optim_list = [optim,optim1,optim2]

# model_list = [model1,model2]
# optim_list = [optim1,optim2]

for m,opti in zip(model_list,optim_list):
  print(f"\n ---------->> TRAINING AND TESTING ON {type(m).__name__} <<---------- \n")
  m = m.to(device)
  m.train()
  best_loss = float("inf")
  for epoch in range(epochs):
    loss_list=[]
    f_loss = 0
    for batch_features,batch_labels in train_loader:
      batch_labels = batch_labels.long()
      batch_features,batch_labels = batch_features.to(device),batch_labels.to(device)
      opti.zero_grad()
      y_pred = m(batch_features)
      loss = loss_fn(y_pred,batch_labels)
      loss.backward()
      opti.step()
      loss_list.append(loss.item())
    f_loss = sum(loss_list)/len(loss_list)
    if(f_loss < best_loss):
      best_loss = f_loss
      m_name = type(m).__name__
      torch.save(m.state_dict(),f"/content/drive/MyDrive/Model_save/{m_name}_{epoch}_{best_loss:.2f}.pth")
      print(f"model saved at :- /content/drive/MyDrive/Model_save/{m_name}_{epoch}_{best_loss:.2f}.pth")
    print(f"Epochs :- {epoch+1},Loss :- {f_loss}")



  # VGG16
  if isinstance(m, models.VGG):
      for param in m.features[-5:].parameters():
          param.requires_grad = True

  # ResNet
  elif isinstance(m, models.ResNet):
      for param in m.layer4.parameters():
         param.requires_grad = True

  # EfficientNet
  elif isinstance(m, models.EfficientNet):
      for param in m.features[-2:].parameters():
          param.requires_grad = True
  opti = torch.optim.Adam(filter(lambda p:p.requires_grad,m.parameters()), lr=0.0001)

  best_loss = float("inf")
  for epoch in range(2):
    loss_list=[]
    f_loss = 0
    for batch_features,batch_labels in train_loader:
      batch_labels = batch_labels.long()
      batch_features,batch_labels = batch_features.to(device),batch_labels.to(device)
      opti.zero_grad()
      y_pred = m(batch_features)
      loss = loss_fn(y_pred,batch_labels)
      loss.backward()
      opti.step()
      loss_list.append(loss.item())
    f_loss = sum(loss_list)/len(loss_list)
    if(f_loss < best_loss):
      best_loss = f_loss
      m_name = type(m).__name__
      torch.save(m.state_dict(),f"/content/drive/MyDrive/Model_save/{m_name}_{epoch}_{best_loss:.2f}.pth")
      print(f"model saved at :- /content/drive/MyDrive/Model_save/{m_name}_{epoch}_{best_loss:.2f}.pth")

    print(f"Epochs :- {epoch+1},Loss :- {f_loss}")

  m.eval()

  correct = 0
  total = 0

  with torch.no_grad():
    for batch_features,batch_labels in test_loader:
      batch_labels = batch_labels.long()
      batch_features,batch_labels = batch_features.to(device),batch_labels.to(device)

      y_pred = m(batch_features)
      y_pred = torch.argmax(y_pred,dim=1)

      correct += (y_pred==batch_labels).sum().item()
      total += batch_labels.size(0)
    accuracy = correct/total
    print("The accuracy on testing dataset is :- ",accuracy)
'''