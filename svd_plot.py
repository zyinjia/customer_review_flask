from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.embed import components
from bokeh.models import FixedTicker, FuncTickFormatter
import matplotlib.pyplot as plt
import pickle
import numpy as np

from topic_model_svd import get_top_words

def get_plotsvd_data(path):
    f = open(path, 'r')
    svd_topic, svd_s = pickle.load(f)
    f.close()
    return svd_topic, svd_s


def plot_bokeh(topics, eigenvalues):
    top_words = get_top_words(topics)
    topic_num = len(topics.keys())
    word_num  = len(top_words)

    X, Y, S, C = [], [], [], []
    for x in topics:
        for item in topics[x]:
            word = item[1]
            radi = item[0]
            X.append(x)
            Y.append(top_words.index(word))
            S.append(abs(radi)**2*40)
            C.append(eigenvalues[x])

    p = figure(title = "Reduce Dimension")
    p.circle(X, Y, fill_color='blue',
            fill_alpha=0.2, size=S)
    p.x_range = Range1d(-1, topic_num)
    p.xaxis.ticker = FixedTicker(ticks=range(topic_num))
    p.xgrid.minor_grid_line_color = 'grey'
    p.xgrid.minor_grid_line_alpha = 0.2
    p.ygrid.minor_grid_line_color = 'grey'
    p.ygrid.minor_grid_line_alpha = 0.2
    p.y_range = Range1d(word_num,-1)
    data = {}
    for i in range(word_num):
        data[i] = str(top_words[i])
    p.yaxis.ticker = FixedTicker(ticks=range(word_num))
    p.yaxis.formatter = FuncTickFormatter(code="""
        var data = %s;
        return data[tick];
        """ %data)
    return p
