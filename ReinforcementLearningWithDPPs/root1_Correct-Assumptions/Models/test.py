import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANN(nn.Module):
    def __init__(self, indim, hidden1, hidden2, outdim):
        super(ANN,self).__init__()
        # 3 layer fully connected NN
        self.inputLayer = nn.Linear(indim,hidden1)
        self.hiddenLayer = nn.Linear(hidden1,hidden2)
        self.outputLayer = nn.Linear(hidden2,outdim)

    def forward(self, x):
        x = self.inputLayer(x)
        x = torch.tanh(x)# activation fn
        x = self.hiddenLayer(x)
        x = torch.tanh(x)
        x = self.outputLayer(x)
        return x

class NN_model:
    ''' Blocker task model of the environment '''
    def __init__(self, new=False):
        ''' I encode the state and action as:
        [r1, c1, r2, c2, r3, c3, dr1, dc1, dr2, dc2, dr3, dc3]
        and the output state as:
        [r1, c1, r2, c2, r3, c3]
        '''
        h1, h2 = 30, 15
        sdim, adim = 6, 6
        self.x_dim, self.y_dim = sdim + adim, sdim

        self.NN = ANN(self.x_dim, h1, h2, self.y_dim).to(device)
        #Define loss criterion
        self.criterion = nn.MSELoss()
        #Define the optimizer
        self.optimiser = torch.optim.Adam(self.NN.parameters(), lr=0.01)

        self.time = 0
        self.training_freq = 1000
        self.batch_size = 50000
        self.buffer_capacity = 200000
        self.memory_buffer = np.zeros((self.buffer_capacity, self.x_dim+self.y_dim))

        self.batch_x = np.zeros((self.buffer_capacity, self.x_dim))
        self.batch_y = np.zeros((self.buffer_capacity, self.y_dim))


        self.path = 'model_states/NN'
        if new==False: self.load()
        
    def save(self):
        torch.save(self.NN.state_dict(), self.path)
    
    def load(self):
        # self.NN = ANN(*args, **kwargs).to(device)
        self.NN.load_state_dict(torch.load(self.path))
        self.NN.eval()


    def encodeState(self, s):
        r, c = 4, 7
        ((r1,c1), (r2,c2), (r3,c3)) = s
        return [r1/r, c1/c, r2/r, c2/c, r3/r, c3/c]

    def encodeAction(self, a):
        ((dr1,dc1), (dr2,dc2), (dr3,dc3)) = a
        return [dr1, dc1, dr2, dc2, dr3, dc3]

    def encode(self, s, a=None):
        if a==None: return np.array(self.encodeState(s))
        x = self.encodeState(s) + self.encodeAction(a)
        return np.array(x)

    def decode(self, y, s):
        r, c = 4, 7
        m = [r, c, r, c, r, c]
        y = np.multiply(m, y)

        fix = lambda mx, i: min(mx, max(int(i), 0))
        [r1, c1, r2, c2, r3, c3] = [fix(mx, i) for mx, i in zip(m, list(np.round(y)))]

        # print(((r1, c1), (r2, c2), (r3, c3)))
        (s1, s2, s3) = ((r1, c1), (r2, c2), (r3, c3))
        if s1==s2 or s1==s3 or s2==s3:
            return s
        return (s1, s2, s3)

    def makeBatch(self):
        mn = min(self.buffer_capacity, self.time)
        idxs = np.random.choice(range(mn), self.batch_size)
        self.batch_x = self.memory_buffer[idxs, :self.x_dim]
        self.batch_y = self.memory_buffer[idxs, :self.y_dim]

    def train(self):
        X = torch.from_numpy(self.batch_x).type(torch.FloatTensor)
        y = torch.from_numpy(self.batch_y).type(torch.FloatTensor)

        y_pred = self.NN.forward(X)
        #Compute MLE loss
        loss = self.criterion(y_pred, y)
        #Clear the previous gradients
        self.optimiser.zero_grad()
        #Compute gradients
        loss.backward()
        #Adjust weights
        self.optimiser.step()

    def update(self, s, a, s_dash, r):
        example = np.array(self.encodeState(s) +
              self.encodeAction(a) + self.encodeState(s_dash))

        index = self.time % self.buffer_capacity
        self.memory_buffer[index] = example

        if self.time % self.training_freq == 0 and self.time > self.batch_size:
            self.makeBatch()
            self.train()

        self.time += 1

    def predict(self, s, a):
        # if self.time < 3000: return s
        x = self.encode(s, a)
        x = torch.from_numpy(x).type(torch.FloatTensor)
        y_pred = self.NN.forward(x)
        return self.decode(y_pred.detach().numpy(), s)





if __name__=="__main__":
    import json
    with open('env_dataset', 'r') as f:
        data = json.load(f)
    n = len(data)

    # model = NN_model(new=True)

    # X = np.zeros((n, 12))
    # Y = np.zeros((n, 6))
    # for i, [s, a, ns, r] in enumerate(data):
    #     x = model.encodeState(s) + model.encodeAction(a)
    #     y = model.encodeState(ns)
    #     X[i, :] = np.array(x)
    #     Y[i, :] = np.array(y)

    # model.batch_x = X
    # model.batch_y = Y

    # ac_time = []

    # def check_acc():
    #     m=400
    #     count = 0
    #     for i in np.random.randint(0,n,m):
    #         [s, a, ns, r] = data[i]
    #         ((r1,c1),(r2,c2),(r3,c3)) = model.predict(s, a)
    #         p = np.array([r1,c1,r2,c2,r3,c3])
    #         ((r1,c1),(r2,c2),(r3,c3)) = ns
    #         y = np.array([r1,c1,r2,c2,r3,c3])
    #         if np.sum(p-y) < 0.01:
    #             count+=1
    #     acc = count/m
    #     ac_time.append(acc)
    #     print('\nACC: '+str(acc))


    # for i in tqdm(range(5000)):
    #     model.train()
    #     if i%10==0:
    #         model.save()
    #         check_acc()
    model = NN_model()

    for [s, a, ns, r] in data[-40:]:
        print('s, a, ns: '+str((s, a, ns))+', pred: '+str(model.predict(s, a)))

    # import matplotlib.pyplot as plt
    # plt.plot(ac_time)
    # plt.show()
    