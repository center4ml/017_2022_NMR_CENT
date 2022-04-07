import math
import matplotlib.pyplot as plt
import torch

frames = 64

TIME = torch.linspace(0., 3 * math.pi, frames)

def generate(samples):
    TARGET = torch.randint(0, 2, (samples,))
    AMPLITUDE0 = 0.2 + 0.8 * torch.rand(samples, 1)
    OMEGA0 = 0.5 + 1.5 * torch.rand(samples, 1)
    PHASE0 = 2. * math.pi * torch.rand(samples, 1)
    SIGNAL0 = AMPLITUDE0 * torch.sin(TIME[:, None, None] * OMEGA0 + PHASE0)
    SIGNAL0 = SIGNAL0.sum(2, keepdim = True)
    AMPLITUDE1 = torch.rand(samples, 2)
    OMEGA1 = 0.5 + 1.5 * torch.rand(samples, 2)
    PHASE1 = 2. * math.pi * torch.rand(samples, 2)
    SIGNAL1 = AMPLITUDE1 * torch.sin(TIME[:, None, None] * OMEGA1 + PHASE1)
    SIGNAL1 = SIGNAL1.sum(2, keepdim = True)
    SIGNAL = torch.where(torch.eq(TARGET, 0)[:, None], SIGNAL0, SIGNAL1)
    return TARGET, SIGNAL

TARGET, SIGNAL = generate(4)

print(TARGET, flush = True)
for sample in range(4):
    plt.plot(TIME, SIGNAL[:, sample])
plt.show()
