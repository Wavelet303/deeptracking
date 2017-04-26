import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.style.use('ggplot')

if __name__ == '__main__':
    path = "/home/mathieu/Dataset/DeepTrack/model/mixed_skull/scores"
    # todo update all files
    name = "mixed_skull15"
    path_csv = os.path.join(path, "{}_eval.csv".format(name))

    fig, axes = plt.subplots(nrows=2, ncols=1)

    df = pd.read_csv(path_csv)
    column_tags = df.columns.values
    df.plot(ax=axes[0], x=np.arange(len(df)), y=column_tags[:3])
    df.plot(ax=axes[1], x=np.arange(len(df)), y=column_tags[3:])
    fig = plt.gcf()
    fig.savefig(os.path.join(path, "{}_plot.png".format(name)))
    #plt.show()

    # Compute scores data
    info = pd.DataFrame()
    info["mean"] = df.mean(axis=0)
    info["std"] = df.std(axis=0)
    info.to_csv(os.path.join(path, "{}_score.csv".format(name)), index=True, encoding='utf-8')
