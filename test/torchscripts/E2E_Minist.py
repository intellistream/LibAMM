import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import CoOccurringFD
# Define your custom matrix multiplication function
def my_matmul(x, y):
    rows,cols=x.shape
    if cols>20:
        sketchSize=cols/10
    else:
        sketchSize=10
    # Your implementation here
    return CoOccurringFD.FDAMM(x,y,int(sketchSize))

# Define a custom Linear layer that uses your custom matrix multiplication function
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def mySqrt(self,a:float):
        y = torch.sqrt(torch.tensor(a, dtype=torch.float32))
        return y.item()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=self.mySqrt(5.0))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / self.mySqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Use your custom matrix multiplication function instead of torch.matmul
        output = my_matmul(input, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output

# Define your neural network architecture
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.relu(self.fc3(x))
        return x
def testNN(net,test_loader):
        #first, load parameters
        pretrained_params = torch.load('pretrained_model.pt')
        custom_params = net.state_dict()

        for name in custom_params:
            if name in pretrained_params:
                custom_params[name] = pretrained_params[name]
        net.load_state_dict(custom_params)
        correct = 0
        total = 0
        #then, run test
        net2=net
        for data in test_loader:
            images, labels = data
            outputs = net2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Accuracy on test set: {correct / total}")
        print(f"Accuracy on test set: {correct / total}")
        return correct / total
def main():
    device='cuda'
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Set up the data loaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the neural network using the default Linear layers
    net = MyNet()

    if os.path.exists('pretrained_model.pt'):
        print('find pretrained model, run test')
        print('first run default version')
        accuracy0=testNN(net,test_loader)
        print('then run coocuuring 1 version')
        # Replace the Linear layers with your custom Linear layers and load the pre-trained weights
        net.fc1 = CustomLinear(784, 128)
        accuracy1=testNN(net,test_loader)
        print('next run coocuuring 2 version')
        net.fc1=nn.Linear(784,128)
        net.fc2 = CustomLinear(128, 128)
        accuracy2=testNN(net,test_loader)
        print('finally run coocuuring 3 version')
        net.fc2 = nn.Linear(128, 128)
        net.fc3 = CustomLinear(128, 10)
        accuracy3=testNN(net,test_loader)
        print('default accuracy=',accuracy0)
        print('co-occuring 1 accuracy=',accuracy1)
        print('co-occuring  2 accuracy=',accuracy2)
        print('co-occuring  3 accuracy=',accuracy3)
        
        
    else:
        print('build pretrain model first')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        net=net.to(device)
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}: loss = {running_loss / len(train_loader)}")
        net=net.to('cpu')
        # Save the pre-trained model
        torch.save(net.state_dict(), 'pretrained_model.pt')

    

    

# Evaluate
if __name__ == '__main__':
    main()