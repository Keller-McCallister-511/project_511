from unicodedata import unidata_version
import torch
from torch import nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def SelfConfidMSELoss(input, target):
    probs = F.softmax(input[0], dim=1)
    confidence = torch.sigmoid(input[1]).squeeze()
    weights = torch.ones_like(target).type(torch.FloatTensor)
    weights[(probs.argmax(dim=1)!=target)] *= 1
    labels_hot = one_hot_embedding(target, 10)

    loss = weights * (confidence - (probs * labels_hot).sum(dim=1)) ** 2
    return torch.mean(loss)

class MLP_Confid(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = True
        self.fc1=nn.Linear(784, 1000)
        self.fc2=nn.Linear(1000, 10)
        self.fc_drop = nn.Dropout(0.3)
        self.uc1=nn.Linear(1000, 400)
        self.uc2=nn.Linear(400,400)
        self.uc3=nn.Linear(400,400)
        self.uc4=nn.Linear(400,400)
        self.uc5=nn.Linear(400,1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        op = x.view(-1, self.fc1.in_features)
        op = F.relu(self.fc1(op))

        if self.dropout:
            op = self.fc_drop(op)
        
        uc = F.relu(self.uc1(op))
        uc = F.relu(self.uc2(uc))
        uc = F.relu(self.uc3(uc))
        uc = F.relu(self.uc4(uc))
        uc = self.uc5(uc)

        op = self.fc2(op)
        pred = self.softmax(op)
        return pred, uc

model = MLP_Confid()
model.load_state_dict(torch.load("../saved_models/mlp_resume.pt"), strict=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#criterion = SelfConfidMSELoss()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=128, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=128, shuffle=True
)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = SelfConfidMSELoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '+
                  f'({(100.*batch_idx/len(train_loader)):.2f}%)]\tLoss: {loss.item():.6f}')
                  
def test(model):
    y_pred=[]
    y_true=[]
    model.eval()
    test_loss = 0
    correct=0

    with torch.no_grad():
        for data, target in test_loader:
            data=data.view(-1,784)
            output=model(data)
            test_loss+=SelfConfidMSELoss(output,target).item()
            pred = output[0].data.max(1, keepdim=True)[1]
            correct+= pred.eq(target.data.view_as(pred)).sum()
            op = output[0]
            op = (torch.max(torch.exp(op),1)[1]).data.cpu().numpy()
            uncertainty = output[1]
            uncertainty = torch.sigmoid(uncertainty)
            y_pred.extend(op)
            target = target.data.cpu().numpy()
            y_true.extend(target)
            output = [op, uncertainty]
            #print(f'Pred: {op}\tUncertainty: {uncertainty}')
            print(output)
    
    test_loss/=len(test_loader.dataset)
    print(f'Test Set: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100.*correct/len(test_loader.dataset)):.2f}%) ')
    conf_mat = confusion_matrix(y_pred, y_true)
    print(conf_mat)

if __name__ == '__main__':
    for epoch in range(1,6):
        train(epoch)
    test(model)
        
    
