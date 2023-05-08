import torch
import numpy as np
import matplotlib.pyplot as plt



# Only steer 2-1
a2c_x_str2 = torch.load('tx_str2.pt').numpy()
a2c_x_str3 = torch.load('tx_str3.pt').numpy()
a2c_x_str3 = 20 + a2c_x_str3
x = np.hstack((a2c_x_str2, a2c_x_str3))
print(x[:40])
print(x.shape)

a2c_y_str2 = torch.load('ty_str2.pt').numpy()
a2c_y_str3 = torch.load('ty_str3.pt').numpy()
y = np.hstack((a2c_y_str2, a2c_y_str3))
print(y.shape)


def plot(x, y, title = "A2C", fn="./a2c.png", shown=False):
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.plot(x, y, color ="red")
    filename = fn
    plt.savefig(filename)
    plt.show()

plot(x, y, title="Only Steer", fn="./a2c_only_steer.png")

#plot(a2c_x_str1, a2c_y_str1, title="A2C STAR", fn="./a2c_star.png")

#plot(a2c_x_str2, a2c_y_str2, title="A2C STAR ONLY STEER 2-1", fn="./a2c_star_only_steer_2-1.png")

#plot(a2c_x_str3, a2c_y_str3, title="A2C STAR ONLY STEER", fn="./a2c_star_only_steer.png")



