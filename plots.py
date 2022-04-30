'''
Generates F1-score plot for the report.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_f1():
    scores = pd.read_csv('scores.csv', index_col=0)
    scores[scores == 0] = np.nan

    for s in scores.index:
        y = scores.loc[s]
        x = np.arange(len(y))
        plt.plot(x, y)
    plt.xticks(x, scores.columns)
    plt.ylabel('F1-score')
    plt.title('F1-score of each method and feature')
    plt.legend(scores.index)
    plt.show()


if __name__ == '__main__':
    plot_f1()
