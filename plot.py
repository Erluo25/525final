import torch
import numpy as np
import matplotlib.pyplot as plt

a2c_x = torch.load('tx.pt').numpy()
a2c_y = torch.load('ty.pt').numpy()

def plot(x, y, title = "A2C", fn="./a2c.png", shown=False):
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.plot(x, y, color ="red")
    filename = fn
    plt.savefig(filename)
    plt.show()

plot(a2c_x, a2c_y)