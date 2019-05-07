import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import interactive


def visualize_attention_weights(attention_weights, sentence: str,
                                cmap='coolwarm'):  # cmap='gist_gray'
    interactive(True)
    # num_heads = np.shape(attention_weights)[0]
    num_heads = 4
    xticklabels = sentence.split(' ')
    yticklabels = xticklabels
    
    f, axes = plt.subplots(int(num_heads / 2), 2)
    
    row = 0
    for i in range(num_heads):
        at11 = attention_weights[i, :len(xticklabels), :len(xticklabels)]
        col = i % 2
        
        sns.heatmap(at11, linewidths=.5, xticklabels=xticklabels,
                    yticklabels=yticklabels, cmap=cmap, cbar=False, ax=axes[row, col])
        if col:
            row += 1
