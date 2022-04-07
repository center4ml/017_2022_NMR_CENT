import math
import matplotlib.pyplot as plt
import torch

frames = 64

TIME = torch.linspace(0., 3 * math.pi, frames, device = 'cuda')

def generate(samples):
    TARGET = torch.randint(0, 2, (samples,), device = 'cuda')
    AMPLITUDE0 = 0.2 + 0.8 * torch.rand(samples, 1, device = 'cuda')
    OMEGA0 = 0.5 + 1.5 * torch.rand(samples, 1, device = 'cuda')
    PHASE0 = 2. * math.pi * torch.rand(samples, 1, device = 'cuda')
    SIGNAL0 = AMPLITUDE0 * torch.sin(TIME[:, None, None] * OMEGA0 + PHASE0)
    SIGNAL0 = SIGNAL0.sum(2, keepdim = True)
    AMPLITUDE1 = torch.rand(samples, 2, device = 'cuda')
    OMEGA1 = 0.5 + 1.5 * torch.rand(samples, 2, device = 'cuda')
    PHASE1 = 2. * math.pi * torch.rand(samples, 2, device = 'cuda')
    SIGNAL1 = AMPLITUDE1 * torch.sin(TIME[:, None, None] * OMEGA1 + PHASE1)
    SIGNAL1 = SIGNAL1.sum(2, keepdim = True)
    SIGNAL = torch.where(torch.eq(TARGET, 0)[:, None], SIGNAL0, SIGNAL1)
    return TARGET, SIGNAL

class Model(torch.nn.Module):
    def __init__(self, size0, size1):
        super(Model, self).__init__()
        self.rnn = torch.nn.GRU(size0, size1)
        self.linear = torch.nn.Linear(size1, 2)
    def forward(self, ACTIVITY0):
        _, ACTIVITY1 = self.rnn(ACTIVITY0)
        ACTIVATION2 = self.linear(ACTIVITY1[0])
        return ACTIVATION2

model = Model(1, 64).cuda()

optimizer = torch.optim.Adam(model.parameters())
for epoch in range(65536):
    TARGET, SIGNAL = generate(256)
    ACTIVATION = model(SIGNAL)
    LOSS = torch.nn.functional.cross_entropy(ACTIVATION, TARGET)
    LABEL = ACTIVATION.argmax(1)
    ACCURACY = torch.eq(LABEL, TARGET).float().mean()
    optimizer.zero_grad()
    LOSS.backward()
    optimizer.step()
    print("%6d %12.4f %12.4f" % (epoch, LOSS, ACCURACY), flush = True)

TARGET, SIGNAL = generate(65536)
ACTIVATION = model(SIGNAL)
LABEL = ACTIVATION.argmax(1)
ACCURACY = torch.eq(LABEL, TARGET).float().mean()
print(ACCURACY.item())

#accuracy: 0.9736
