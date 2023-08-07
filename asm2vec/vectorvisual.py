import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

'''
本模块对词嵌入得到的向量进行可视化操作
'''

def tsne():
    dataset = np.genfromtxt(fname='./data/w2vvectors.tsv', dtype=None)
    dataset2 = np.genfromtxt(fname='./data/lstmvectors.tsv', dtype=None)
    show_plt(dataset)
    show_plt(dataset2)


def show_plt(dataset):
    t_sne = TSNE(
        # Number of dimensions to display. 3d is also possible.
        n_components=2,
        # Control the shape of the projection. Higher values create more
        # distinct but also more collapsed clusters. Can be in 5-50.
        perplexity=5,
        init="random",
        verbose=1,
        square_distances=True,
        learning_rate="auto").fit_transform(dataset)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.grid(False)
    plt.scatter(t_sne[:, 0], t_sne[:, 1])
    metadata_file = open('../data/lstmmetadata.tsv')
    metadatas = metadata_file.readlines()
    metadata_file.close()
    for i in range(len(t_sne[:, 0]) - 1):
        plt.annotate(metadatas[i+1], xy=(t_sne[:, 0][i+1], t_sne[:, 1][i+1]))
    plt.show()


if __name__ == '__main__':
    tsne()
