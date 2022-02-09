from cProfile import label
from os import name
import matplotlib.pyplot as plt


def plot(data1, data2, title, idx):

    plt.figure(idx)
    plt.title(title)
    plt.ylabel("accuracy")  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(data1,
             label='Cubic',
             marker="o",
             linestyle="-")
    plt.plot(data2, label='Linear', marker="o", linestyle="-")

    if idx == 1:
        plt.plot(gradually_overlap, label='Gradually Freezing',
                 marker="o", linestyle="-")
    elif idx == 2:
        plt.plot(gradually_loss, label='Gradually Freezing',
                 marker="o", linestyle="-")
    plt.legend()


def show():
    plt.show()


def offline_plot(idx, title, y_title):
    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_title)  # y label
    plt.xlabel("Epochs")  # x label

    if idx == 1:
        plt.plot(freezeout_cubic_acc,
                 label='Cubic',
                 marker="o",
                 linestyle="-")
        plt.plot(freezeout_linear_acc, label='Linear',
                 marker="o", linestyle="-")
        plt.plot(gradually_overlap, label='Gradually Freezing',
                 marker="o", linestyle="-")
    elif idx == 2:
        plt.plot(freezeout_cubic_loss,
                 label='Cubic',
                 marker="o",
                 linestyle="-")
        plt.plot(freezeout_linear_loss, label='Linear',
                 marker="o", linestyle="-")
        plt.plot(gradually_loss, label='Gradually Freezing',
                 marker="o", linestyle="-")
    plt.legend()


freezeout_cubic_acc = [0.3136, 0.2784, 0.4414, 0.3892, 0.4636, 0.4918, 0.5384,
                       0.5414, 0.5489999999999999, 0.5511999999999999, 0.5878, 0.5888]

freezeout_linear_acc = [0.14180000000000004, 0.30500000000000005, 0.3284, 0.38539999999999996,
                        0.5184, 0.45840000000000003, 0.5758, 0.5934, 0.6154, 0.618, 0.617, 0.6186]

gradually_overlap = [0.3580000102519989, 0.37940001487731934, 0.46790000796318054, 0.48840001225471497, 0.5077999830245972,
                     0.5200999975204468, 0.5306000113487244, 0.5350000262260437, 0.5404000282287598, 0.5449000000953674, 0.5491999983787537, 0.5543000102043152]

freezeout_cubic_loss = [1.8781439699704134, 2.0998384786557547, 1.5562068299402165, 1.852144894720633, 1.5108785116219823,
                        1.4380403530748584, 1.3554465453835982, 1.3331090634382223, 1.321497917175293, 1.312872596933872, 1.2123649059971677, 1.2102767998659159]
freezeout_linear_loss = [2.719791059252582, 2.1698374536973013, 1.9696003364611276, 1.7617521044574207, 1.3903245986262454,
                         1.5947002097021175, 1.2138738081425051, 1.179070402549792, 1.1184102960779696, 1.1014134748072564, 1.1026951112324679, 1.1021680386760566]

gradually_loss = [1.841166377067566, 1.7962379455566406, 1.5106322765350342, 1.4514060020446777, 1.400439739227295, 1.3723630905151367,
                  1.3482745885849, 1.3268492221832275, 1.3107562065124512, 1.301764726638794, 1.2902268171310425, 1.2782020568847656]


if __name__ == '__main__':
    offline_plot(1, "FreezeOut overlap Accuracy", "accuracy")
    offline_plot(2, "FreezeOut overlap loss", "loss")
    show()
