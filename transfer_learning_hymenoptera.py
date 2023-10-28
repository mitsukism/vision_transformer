import torch
from torch import nn
import timm
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vit_tiny_patch16_224', type=str, help='model name')
parser.add_argument('--checkpoint', default='/content/drive/MyDrive/ch3/ImageNet/tiny16/best_checkpoint.pth', type=str, help='checkpoint')
parser.add_argument('--num_epochs', default=1, type=int, help='number of classes')
args = parser.parse_args()

model = timm.create_model(args.model, pretrained=False)
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(checkpoint["model"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

params_to_update = []
update_param_names = ['head.weight', 'head.bias']

for name, param in model.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder(root='/content/hymenoptera_data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='/content/htmenoptera_data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 損失関数とオプティマイザを定義
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 転移学習
num_epochs = args.num_epochs  # argparseを使ってエポック数を取得

for epoch in range(num_epochs):
    train_total_loss = 0.0
    train_correct = 0
    train_total = 0

    val_total_loss = 0.0
    val_correct = 0
    val_total = 0

    # 訓練フェーズ
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        _, train_predicted = torch.argmax(outputs, axis=1)
        train_total += labels.size(0)
        train_correct += (train_predicted == labels).sum().item()

    train_avg_loss = train_total_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    # 検証フェーズ
    for i, (data, labels) in enumerate(val_loader):
        data, labels = data.to(device), labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        val_total_loss += loss.item()
        _, val_predicted = torch.argmax(outputs, axis=1)
        val_total += labels.size(0)
        val_correct += (val_predicted == labels).sum().item()

    val_avg_loss = val_total_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Average Loss: {train_avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Average Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
