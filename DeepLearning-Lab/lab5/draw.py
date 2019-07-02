import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio

# 使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 画散点图
def draw_scatter(data, color, x_min, x_max, y_min, y_max):
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Scatter Plot')
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=10)

# 画背景
def draw_background(D, x_min, x_max, y_min, y_max):
    i = x_min
    bg = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            bg.append([i, j])    
            j += 0.01
        bg.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        bg.append([i, j])    
        j += 0.01
        bg.append([i, y_max])
    bg.append([x_max, y_max])
    color = D(torch.Tensor(bg).to(device))
    bg = np.array(bg)
    cm = plt.cm.get_cmap('gray')
    sc = plt.scatter(bg[:, 0], bg[:, 1], c= np.squeeze(color.cpu().data), cmap=cm)
    # 显示颜色等级
    cb = plt.colorbar(sc)
    return cb

# 合成动图
def img_show(net):
    imgs = []
    for i in range(1, 21):
        fileName = "./result/" + net + "/epoch" + str(i*5) + ".png"
        imgs.append(imageio.imread(fileName))
    imageio.mimsave("./result/" + net + "/final.gif", imgs, fps=1)

if __name__ == '__main__':
    img_show("gan")
    img_show("wgan")
    img_show("wgan_gp")

