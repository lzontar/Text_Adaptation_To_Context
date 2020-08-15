import matplotlib.pyplot as plt
import seaborn as sns

color_SOC_MED = 'salmon'
color_NEWS = 'gold'
color_ABSTR = 'dodgerblue'
color_OFC_STMT = 'lightgreen'

def vis_mean_metrics(text_metrics, variance_metrics):
    x = ['Official statements', 'Research articles', 'News', 'Social media']
    length = [text_metrics['LEN']['OFC_STMT'], text_metrics['LEN']['RES_ARTCL'], text_metrics['LEN']['NEWS'], text_metrics['LEN']['SOC_MED']]
    polarity = [text_metrics['SENT_ANAL']['OFC_STMT']['POLAR'], text_metrics['SENT_ANAL']['RES_ARTCL']['POLAR'], text_metrics['SENT_ANAL']['NEWS']['POLAR'], text_metrics['SENT_ANAL']['SOC_MED']['POLAR']]
    readability = [text_metrics['READ']['OFC_STMT'], text_metrics['READ']['RES_ARTCL'], text_metrics['READ']['NEWS'], text_metrics['READ']['SOC_MED']]

    plt.style.use('ggplot')

    variance_length = [variance_metrics['LEN']['OFC_STMT'], variance_metrics['LEN']['RES_ARTCL'], variance_metrics['LEN']['NEWS'],
              variance_metrics['LEN']['SOC_MED']]
    variance_polarity = [variance_metrics['SENT_ANAL']['OFC_STMT'], variance_metrics['SENT_ANAL']['RES_ARTCL'],
                variance_metrics['SENT_ANAL']['NEWS'], variance_metrics['SENT_ANAL']['SOC_MED']]
    variance_readability = [variance_metrics['READ']['OFC_STMT'], variance_metrics['READ']['RES_ARTCL'], variance_metrics['READ']['NEWS'],
                   variance_metrics['READ']['SOC_MED']]

    x_pos = [i for i, _ in enumerate(x)]

    barlist = plt.barh(x_pos, length, xerr=variance_length)

    barlist[0].set_color(color_OFC_STMT)
    barlist[1].set_color(color_ABSTR)
    barlist[2].set_color(color_NEWS)
    barlist[3].set_color(color_SOC_MED)

    plt.ylabel("Publication type")
    plt.xlabel("Length (number of tokens)")
    # plt.title("Initial values of length for different publicaiton types")

    plt.yticks(x_pos, x)

    for i in range(len(x_pos)):
        plt.annotate(str(length[i]), xy=(length[i] + 100, x_pos[i] + 0.05))

    plt.show()
    plt.clf()

    barlist = plt.barh(x_pos, readability, xerr=variance_readability)

    barlist[0].set_color(color_OFC_STMT)
    barlist[1].set_color(color_ABSTR)
    barlist[2].set_color(color_NEWS)
    barlist[3].set_color(color_SOC_MED)

    plt.ylabel("Publication type")
    plt.xlabel("Flesch Reading Ease score")
    # plt.title("Initial values of readability score for different publicaiton types")

    plt.yticks(x_pos, x)
    for i in range(len(x_pos)):
        plt.annotate(str(readability[i]), xy=(readability[i] + 1, x_pos[i] + 0.05))

    plt.show()
    plt.clf()

    barlist = plt.barh(x_pos, polarity, xerr=variance_polarity)

    barlist[0].set_color(color_OFC_STMT)
    barlist[1].set_color(color_ABSTR)
    barlist[2].set_color(color_NEWS)
    barlist[3].set_color(color_SOC_MED)

    plt.ylabel("Publication type")
    plt.xlabel("Polarity score")
    # plt.title("Initial values of polarity score for different publicaiton types")

    plt.yticks(x_pos, x)

    for i in range(len(x_pos)):
        plt.annotate(str(polarity[i]), xy=(polarity[i] + 0.005, x_pos[i] + 0.05))

    plt.show()
    plt.clf()

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime

def vis_results(initial_values, results, target, target_value):

    labels = []
    colors = []
    target_color = None
    if target == 'OFC_STMT':
        labels = ['Research articles', 'News', 'Social media']
        colors = [color_ABSTR, color_NEWS, color_SOC_MED]
        target_color = color_OFC_STMT
    elif target == 'RES_ARCL':
        labels = ['Official statements', 'News', 'Social media']
        colors = [color_OFC_STMT, color_NEWS, color_SOC_MED]
        target_color = color_ABSTR
    elif target == 'NEWS':
        labels = ['Official statements', 'Research articles', 'Social media']
        colors = [color_OFC_STMT, color_ABSTR, color_SOC_MED]
        target_color = color_NEWS
    elif target == 'SOC_MED':
        labels = ['Official statements', 'Research articles', 'News', ]
        colors = [color_OFC_STMT, color_ABSTR, color_NEWS]
        target_color = color_SOC_MED


    import matplotlib.pyplot as plt
    import numpy as np

    men_means = [20, 34, 30]
    women_means = [25, 32, 34]

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label='Initial value', color='lightskyblue')
    rects2 = ax.bar(x + width / 2, women_means, width, label='Result', color='lightslategray')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.axhline(y=target_value, color='black', linestyle='-', label='Target value')
    ax.legend()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()