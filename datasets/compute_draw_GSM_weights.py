import numpy as np
import matplotlib.pyplot as plt
import matplotlib

category_names = ['Forward Feature', 'Backward Feature',
                'Frequency Feature']
results = {}

def calculate_average(data):
    """
    计算给定数据列表的平均值
    :param data: 数据列表，每个元素包含三个数字
    :return: 三个数字的均值
    """
    num_values = len(data)
    if num_values == 0:
        return None
    avg_data = [sum(x) / num_values for x in zip(*data)]
    return avg_data


def visualize_data(input_file):
    image_data = []
    bar_data = []
    labels = []

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith('.gz'):
                if image_data:
                    avg_data = calculate_average(image_data)
                    if avg_data:
                        bar_data.append(avg_data)
                        labels.append('Scan ' + str(len(labels)+1))
                    image_data = []
            else:
                values = line.split('\t')
                if len(values) == 3:
                    image_data.append([float(val) for val in values])

    # 检查最后一组图像数据
    if image_data:
        avg_data = calculate_average(image_data)
        if avg_data:
            bar_data.append(avg_data)
            labels.append("最后一组")

    # 绘制图像
    num_images = len(bar_data)
    # num_bars = len(bar_data[0])
    print(bar_data)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # bar_width = 0.6 / num_bars
    # index = np.arange(num_images)

    # for i in range(num_bars):
    #     bar_values = [data[i] for data in bar_data]
    #     ax.barh(index, bar_values, bar_width, left=i * bar_width, label=f"Number {i+1}")

    # ax.set_yticks(index)
    # ax.set_yticklabels(labels)
    # ax.set_xlabel('Proportion')
    # ax.set_title('Image Data Visualization')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('GSM_data_visualization.png')

    # plt.show()
    for i in range(num_images):
        results[labels[i]] = bar_data[i]

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    # category_colors = plt.colormaps['RdYlGn'](
    #     np.linspace(0.15, 0.85, data.shape[1]))
    category_colors = np.array([[0.89888504,0.30549789,0.20676663,1],
                                [0.99315648, 0.73233372, 0.42237601, 1],
                                [0.24805844,0.66720492,0.3502499,1]])
    print(category_colors)
    fig, ax = plt.subplots(figsize=(9.2, 6))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    ax.tick_params(axis='y', labelsize=9)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]

        widths = [round(num, 3) for num in widths]
        print(widths)
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.8,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'black' if r * g * b < 0.5 else 'darkgrey' # white darkgrey
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='medium') # 'lower left'
    ax.margins(y=0.01)
    return fig, ax


# 输入文件路径
input_file = 'TMamba3D_GSM_weights.txt'
visualize_data(input_file)
survey(results, category_names)
plt.tight_layout()
plt.savefig('GSM_data_visualization_all_stages.pdf', format='pdf', bbox_inches='tight')

