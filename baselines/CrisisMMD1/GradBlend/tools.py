import matplotlib.pyplot as plt
import numpy as np

def drawGraph(inputs, title, y_label, legend_name, save_path):
    for item in inputs:
        plt.plot(item)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.legend(legend_name, loc='upper left')
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    return 0

def drawScatters(outputs, labels, save_path = "./"):
    x = []
    y = []
    plt.title("visualization of the fc layer")
    for j in range(len(outputs)):
        x.append(outputs[j][0])
        y.append(outputs[j][1])

    makers = ['<', '.']
    for i in [0,1]:
        idx_pre = [lb==i for lb in labels]
        x_scatter = [i for i,j in zip(x,idx_pre) if j]
        y_scatter = [i for i,j in zip(y,idx_pre) if j]
        plt.scatter(x_scatter, y_scatter, marker = makers[i])
    plt.savefig(save_path)
    plt.show()
    plt.clf()
    return 0


def method(num, mtd):
    return 1/np.sqrt(num) if mtd=="isns" else 1/num

def weight_caculator(class_dist, mtd="ins"):
    Z = 0
    weights = []
    for idx, (key, value) in enumerate(class_dist.items()):
        this_z = method(value, mtd)
        weights.append(this_z)
        Z += this_z
    weights_norm = [float("{:.2f}".format(x/Z)) for x in weights]

    return weights_norm


def datasetConfig(x):
    if x =="crisismmd":
        return "../datasets/task1/"
    elif x =="mvsa":
        return "../datasets/mvsa_data/5fold"
