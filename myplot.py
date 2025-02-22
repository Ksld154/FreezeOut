import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot(data1, data2, y_label, title, idx):
    ax1 = plt.figure(idx).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_label)  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(data1,
             label='t1',
             marker="o",
             linestyle="-")
    plt.plot(data2, label='t2', marker="o", linestyle="-")

    # if idx == 1:
    #     plt.plot(gradually_acc, label='Gradually Freezing',
    #              marker="o", linestyle="-")
    # elif idx == 2:
    #     plt.plot(gradually_loss, label='Gradually Freezing',
    #              marker="o", linestyle="-")
    plt.legend()

def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Epochs")  # x label

    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend(loc='lower right')

def show():
    plt.show()

def save_figure(filepath:str):
    plt.savefig(filepath)


def offline_plot(idx, title, y_title):
    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_title)  # y label
    plt.xlabel("Epochs")  # x label

    if idx == 1:
        plt.plot(baseline_1,
                 label='Baseline: FreezeOut t_0=0.3',
                 marker="o",
                 linestyle="--")
        plt.plot(baseline_2, label='Baseline: FreezeOut t_0=0.8',
                 marker="o", linestyle="--")

        plt.plot(switch_t0_03, label='Switch: FreezeOut t_0=0.3',
                 marker="o", linestyle="-")
        plt.plot(switch_t0_08, label='Switch: FreezeOut t_0=0.8',
                 marker="o", linestyle="-")
        # plt.plot(gradually_overlap, label='Gradually Freezing',
        #          marker="o", linestyle="-")
    elif idx == 2:
        plt.plot(freezeout_cubic_loss,
                 label='Cubic',
                 marker="o",
                 linestyle="-")
        plt.plot(freezeout_linear_loss, label='Linear',
                 marker="o", linestyle="-")
        # plt.plot(gradually_loss, label='Gradually Freezing',
        #          marker="o", linestyle="-")
    plt.legend()


baseline_1 = [0.20540000000000003, 0.20319999999999994, 0.2631999999999999, 0.26760000000000006, 0.34840000000000004, 0.31299999999999994, 0.3558, 0.3526, 0.36419999999999997, 0.38639999999999997, 0.369, 0.391, 0.39239999999999997, 0.39560000000000006, 0.40180000000000005, 0.40859999999999996, 0.40380000000000005, 0.40180000000000005, 0.3964, 0.3992, 0.41679999999999995, 0.4152, 0.40759999999999996, 0.40959999999999996, 0.4094, 0.41259999999999997, 0.41859999999999997, 0.4184, 0.4142, 0.41880000000000006, 0.4152, 0.42000000000000004, 0.42079999999999995, 0.4214, 0.4196, 0.41759999999999997, 0.41780000000000006, 0.4192, 0.41759999999999997, 0.4196, 0.4204, 0.4196, 0.4182, 0.42079999999999995, 0.4182, 0.41979999999999995, 0.4202, 0.41979999999999995, 0.41979999999999995, 0.41979999999999995]
baseline_2 = [0.1754000000000001, 0.2809999999999999, 0.32499999999999996, 0.2734000000000001, 0.479, 0.45499999999999996, 0.44220000000000004, 0.4312, 0.483, 0.5122, 0.5262, 0.43799999999999994, 0.5174000000000001, 0.5992, 0.5318, 0.4798, 0.609, 0.6174, 0.6166, 0.634, 0.6242000000000001, 0.6517999999999999, 0.629, 0.6656, 0.6328, 0.662, 0.6622, 0.6524000000000001, 0.6628000000000001, 0.6738, 0.6796, 0.683, 0.6828000000000001, 0.6821999999999999, 0.6859999999999999, 0.6834, 0.6874, 0.6872, 0.6872, 0.6856, 0.6894, 0.6894, 0.6878, 0.6864, 0.6864, 0.6886, 0.6881999999999999, 0.6881999999999999, 0.6876, 0.6878]

switch_t0_03 = [0.241, 0.28680000000000005, 0.19819999999999993, 0.3373999999999999, 0.3468000000000001, 0.34140000000000004, 0.3842, 0.38380000000000003, 0.36719999999999997, 0.38739999999999997, 0.3758, 0.4022, 0.41259999999999997, 0.4052, 0.4132, 0.4054, 0.41900000000000004, 0.4152, 0.39959999999999996, 0.41079999999999994, 0.42399999999999993, 0.4256, 0.4314, 0.42380000000000007, 0.4282, 0.4214, 0.4266, 0.42299999999999993, 0.42479999999999996, 0.6686, 0.6706000000000001, 0.6699999999999999, 0.6734, 0.6714, 0.6698, 0.6776, 0.6746000000000001, 0.6736, 0.6728000000000001, 0.671, 0.6736, 0.6738, 0.6738, 0.6754, 0.6746000000000001, 0.6754, 0.6748, 0.6748, 0.675, 0.6752]
switch_t0_08 = [0.18220000000000003, 0.3698, 0.30379999999999996, 0.3184, 0.49439999999999995, 0.4478, 0.41579999999999995, 0.49639999999999995, 0.4062, 0.45399999999999996, 0.512, 0.43920000000000003, 0.5247999999999999, 0.5956, 0.521, 0.5574, 0.6044, 0.5912, 0.625, 0.62, 0.6254, 0.6346, 0.6212, 0.659, 0.6354, 0.65, 0.6506000000000001, 0.6617999999999999, 0.6557999999999999, 0.6626, 0.6672, 0.6661999999999999, 0.6696, 0.6712, 0.673, 0.6739999999999999, 0.6732, 0.6728000000000001, 0.6742, 0.6742, 0.6742, 0.6734, 0.6734, 0.672, 0.6739999999999999, 0.6734, 0.6734, 0.6732, 0.6736, 0.6738]


if __name__ == '__main__':
    offline_plot(1, "FreezeOut w/ Switch Model Accuracy", "Accuracy")
    # offline_plot(2, "FreezeOut overlap loss", "loss")
    show()
