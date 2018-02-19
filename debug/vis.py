import matplotlib.pyplot as plt
import seaborn as sns


def attention_visualization(att):
    sns.heatmap(att[0, :, :])
    plt.show()
