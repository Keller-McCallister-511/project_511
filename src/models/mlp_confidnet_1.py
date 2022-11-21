import argparse
import torch
from torch import nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, average_precision_score
from collections import OrderedDict
import numpy as np

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

def summary(results):
    correct = {}
    miss = {}
    good = 0
    bad = 0
    for i in results:
        for x in range(0, len(i[0])):
            if i[0][x] == i[2][x]:
                if i[0][x] in correct:
                    correct[i[0][x]].append(i[1][x].item())
                else:
                    correct[i[0][x]]=[i[1][x].item()]
                good += 1
            else:
                if i[0][x] in miss:
                    miss[i[0][x]].append(i[1][x].item())
                else:
                    miss[i[0][x]]=[i[1][x].item()]
                bad += 1
    for key in correct:
        correct[key] = format((sum(correct[key])/len(correct[key]))*100., ".2f") + '%'
    for key in miss:
        miss[key] = format((sum(miss[key])/len(miss[key]))*100., ".2f") + '%'
    correct = OrderedDict(sorted(correct.items()))
    miss = OrderedDict(sorted(miss.items()))
    print('\t Correct\tIncorrect')
    for key in correct:
        print(f'Class {key}: {correct[key]}\t  {miss[key]}')
    print(good)
    print(bad)

class MLP_Confid(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = True
        self.fc1=nn.Linear(784, 1000)
        self.fc2=nn.Linear(1000, 256)
        self.fc3=nn.Linear(256, 512)
        self.fc4=nn.Linear(512,10)
        self.fc_drop = nn.Dropout(0.3)
        self.uc1=nn.Linear(1000, 400)
        self.uc2=nn.Linear(400,400)
        self.uc3=nn.Linear(400,400)
        self.uc4=nn.Linear(400,400)
        self.uc5=nn.Linear(400,1)

    def forward(self, x):
        op = x.view(-1, self.fc1.in_features)
        op1 = F.relu(self.fc1(op))
        op = F.relu(self.fc2(op1))
        op = F.relu(self.fc3(op))

        if self.dropout:
            op = self.fc_drop(op)
        
        uc = F.relu(self.uc1(op1))
        uc = F.relu(self.uc2(uc))
        uc = F.relu(self.uc3(uc))
        uc = F.relu(self.uc4(uc))
        uc = self.uc5(uc)

        pred = self.fc4(op)
        return pred, uc

model = MLP_Confid()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
    results=[]
    acc, err, proba_pred = [],[],[]
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
            op = (torch.max(op,1)[1]).data.cpu().numpy()
            uncertainty = output[1]
            uncertainty = torch.sigmoid(uncertainty)
            y_pred.extend(op)
            t = target.data.cpu().numpy()
            y_true.extend(target)
            output = [op, uncertainty, t]
            results.append(output)
            acc.extend(pred.eq(target.view_as(pred)).detach().to("cpu").numpy())
            err.extend((pred != target.view_as(pred)).detach().to("cpu").numpy())
            proba_pred.extend(uncertainty.detach().to("cpu").numpy())
            
            
    
    test_loss/=len(test_loader.dataset)
    print(f'Test Set: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100.*correct/len(test_loader.dataset)):.2f}%) ')
    conf_mat = confusion_matrix(y_pred, y_true)
    print(conf_mat)
    summary(results)
    acc = np.reshape(acc, newshape=(len(acc), -1)).flatten()
    err = np.reshape(err, newshape=(len(err), -1)).flatten()
    proba_pred = np.reshape(proba_pred, newshape=(len(proba_pred), -1)).flatten()
    ap_errors = average_precision_score(err, -(proba_pred))
    ap_succ = average_precision_score(acc, (proba_pred))
    print(f'{(ap_errors):05.2%}')
    print(f'{(ap_succ):05.2%}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP_0")
    parser.add_argument('--epochs', type=int, default=25, metavar='N', help='number of epochs for training (default: 25)')
    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--test', action='store_true', default=False, help='test model')
    args=parser.parse_args()

    if args.train:
        for epoch in range(1, args.epochs+1):
            model.load_state_dict(torch.load("../saved_models/mlp_1_resume.pt"), strict=False)
            train(epoch)
            torch.save(model.state_dict(), f"../saved_models/mlp_confidence_1.pt")
    
    if args.test:
        model.load_state_dict(torch.load('../saved_models/mlp_confidence_1.pt'))
        test(model)