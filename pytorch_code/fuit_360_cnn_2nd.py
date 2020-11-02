from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torchvision.models as models

learning_rate = 0.001
training_epochs = 20
batch_size = 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root="/nfs/home/seonbinara/fruits-360/Training", transform=trans)
testset = torchvision.datasets.ImageFolder(root="/nfs/home/seonbinara/fruits-360/Test", transform=trans)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# CNN Model
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 100, 100, 1)
        #    Conv     -> (?, 100, 100, 16)
        #    Pool     -> (?, 50, 50, 16)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 50, 50, 16)
        #    Conv      ->(?, 50, 50, 32)
        #    Pool      ->(?, 25, 25, 32)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 25, 25, 32)
        #    Conv      ->(?, 25, 25, 64)
        #    Pool      ->(?, 12, 12, 64)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L4 ImgIn shape=(?, 12, 12, 64)
        #    Conv      ->(?, 12, 12, 128)
        #    Pool      ->(?, 6, 6, 128)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L5 FC 6x6x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(6 * 6 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L6 Final FC 625 inputs -> 131 outputs
        self.fc2 = torch.nn.Linear(625, 131, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer5(out)
        out = self.fc2(out)
        return out

model = CNN().to(device)

PATH="/nfs/home/seonbinara/fruit_cnn.pth"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_data = len(trainset) # 67692
iteration_num = len(trainloader)

print("LEARNING STARTS! (model: normalCNN)")
print("total data is ", total_data)
print("there will be about ", iteration_num, "steps")

current_accuracy = 0
for epoch in range(training_epochs):
    model.train()
    avg_cost = 0
    step = 0
    for X, Y in trainloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / iteration_num
        if(step % 10 == 0):
            print("STEP [", step, "/", iteration_num, "] LOSS: ", cost.item())
        step += 1
    print('[EPOCH: {:>4}] COST = {:>.9}'.format(epoch + 1, avg_cost))
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # set the model to evaluation mode (dropout=False)

        for images, labels in testloader:
            outputs = model(images.to(device))
            total += labels.size(0)
            correct_prediction = torch.argmax(outputs.data, 1) == labels.to(device)
            correct += correct_prediction.sum()

        accuracy = int(correct) / total
        print('EPOCH', epoch+1,  ' ACCURACY:', accuracy)
        if(accuracy > current_accuracy):
            print('IMPROVEMENT WAS THERE. SAVE CKPT...')
            PATH = "/nfs/home/seonbinara/fruit_cnn.pth"
            torch.save(model.state_dict(), PATH)
            current_accuracy = accuracy


print("LEARNING FINISHED! (model: normalCNN)")
print('FINAL ACCURACY: ', current_accuracy)