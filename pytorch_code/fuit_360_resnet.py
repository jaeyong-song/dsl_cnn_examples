import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt

learning_rate = 0.001
momentum = 0.9
training_epochs = 20
batch_size = 512


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root="/nfs/home/seonbinara/fruits-360/Training", transform=trans)
testset = torchvision.datasets.ImageFolder(root="/nfs/home/seonbinara/fruits-360/Test", transform=trans)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

resnet50 = models.resnet50(pretrained=True).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(resnet50.parameters(), momentum=momentum, lr=learning_rate)

total_data = len(trainset) # 67692
iteration_num = len(trainloader)

print("LEARNING STARTS! (model: Resnet50)")
print("total data is ", total_data)
print("there will be about ", iteration_num, "steps")

current_accuracy = 0
for epoch in range(training_epochs):
    resnet50.train()
    avg_cost = 0
    step = 0
    for X, Y in trainloader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = resnet50(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / iteration_num
        if(step % 40 == 0):
            print("STEP [", step, "/", iteration_num, "] LOSS: ", cost.item())
        step += 1
    print('[EPOCH: {:>4}] COST = {:>.9}'.format(epoch + 1, avg_cost))
    correct = 0
    total = 0
    with torch.no_grad():
        resnet50.eval()  # set the model to evaluation mode (dropout=False)

        for images, labels in testloader:
            outputs = resnet50(images.to(device))
            total += labels.size(0)
            correct_prediction = torch.argmax(outputs.data, 1) == labels.to(device)
            correct += correct_prediction.sum()

        accuracy = int(correct) / total
        print('EPOCH', epoch+1,  ' ACCURACY:', accuracy)
        if(accuracy > current_accuracy):
            print('IMPROVEMENT WAS THERE. SAVE CKPT...')
            PATH = "/nfs/home/seonbinara/fruit_resnet.pth"
            torch.save(resnet50.state_dict(), PATH)
            current_accuracy = accuracy


print("LEARNING FINISHED! (model: Resnet50)")
print('FINAL ACCURACY: ', current_accuracy)