import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

if __name__ == '__main__':
    path = "../train_test_nogit/eval_diff.csv"

    df = pd.read_csv(path)
    column_tags = df.columns.values
    df.plot(x=np.arange(len(df)), y=column_tags[:3])
    plt.show()
    df.plot(x=np.arange(len(df)), y=column_tags[3:])
    plt.show()