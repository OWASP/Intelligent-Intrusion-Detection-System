#neural net classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, in_dim =1, n_class =23):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(144, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
class RNN(nn.Module):

    def __init__(self, input_dim =1, hidden_dim= 3, layer_dim =3, output_dim=1):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
            
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SOM(nn.Module):

    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m*n, dim)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: np.linalg.norm(vect-self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):
        dists = self.pdist(torch.stack([x for i in range(self.m*self.n)]), self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index,:]
        bmu_loc = bmu_loc.squeeze()


# Neural network parameters " would be collected from JSON Config"
batch_size = 128
learning_rate = 1e-2
num_epoches = 5
USE_GPU = torch.cuda.is_available()

model = CNN(1, 23) #would be set in config
#model = SOM(1, 23)
#model = RNN(1, 23)
#model = AE(1, 23)

def neural_train():
    
    global model

    if USE_GPU:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(dataset.train_dataloader, 1):
            img, label = data
            if USE_GPU:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # Spread forward
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
           # Spread backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(dataset.train_dataset)), running_acc / (len(
                dataset.train_dataset))))
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in dataset.test_dataloader:
            img, label = data
            if USE_GPU:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            dataset.test_dataset)), eval_acc / (len(dataset.test_dataset))))
        torch.save(model, filepath)
        print()
