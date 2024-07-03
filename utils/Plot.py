import numpy as np
import matplotlib.pyplot as plt


def plot_orbit(positions: list):
    '''
    :param positions: position vector 리스트
    :return: 위성궤도 3차원 이미지 출력
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)
    ax.set_zlim(-10000, 10000)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('auto')
    ax.grid(True)
    positions = np.array(positions)

    # draw earth sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6378 * np.outer(np.cos(u), np.sin(v))
    y = 6378 * np.outer(np.sin(u), np.sin(v))
    z = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()


def format_dates(dates):
    return [f"{date[0]}-{date[1]}-{date[2]}" for date in dates]


def plot_vertical_lines(dates, formatted_dates):
    for i, date in enumerate(formatted_dates):
        if date == "2024-2-19" or date == "2024-2-24":
            plt.axvline(x=i, color="red", linestyle="--")


def set_plot_params(title, ylabel):
    plt.title(title)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(rotation=30, fontsize=12)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))


def calculate_ylim(data):
    ylim_min = min(data) - (max(data) - min(data)) * 0.1
    ylim_max = max(data) + (max(data) - min(data)) * 0.1
    return ylim_min, ylim_max


def draw_graph(data, title:str, ylabel, color="blue"):
    plt.figure(figsize=(10, 5))
    # formatted_dates = format_dates(data.keys())
    plt.plot(data.index, data.values, color=color)
    # plot_vertical_lines(data.keys(), formatted_dates)
    set_plot_params(title, ylabel)
    ylim_min, ylim_max = calculate_ylim(list(data.values))
    plt.ylim(ylim_min, ylim_max)
    plt.show()


def draw_altitudes(apogees, perigees, average_altitudes):
    plt.figure(figsize=(10, 5))
    formatted_dates = format_dates(apogees.keys())
    plt.plot(formatted_dates, apogees.values(), color="blue", label="apogee")
    plt.plot(formatted_dates, perigees.values(), color="red", label="perigee")
    plt.plot(formatted_dates, average_altitudes.values(), color="green", label="average altitude")
    plt.legend()
    all_data = list(apogees.values()) + list(perigees.values()) + list(average_altitudes.values())
    ylim_min = min(all_data) - (max(all_data) - min(all_data)) * 0.2
    ylim_max = max(all_data) + (max(all_data) - min(all_data)) * 0.2
    plt.ylim(ylim_min, ylim_max)
    plot_vertical_lines(apogees.keys(), formatted_dates)
    set_plot_params("Orbit Height", "Altitude")
    plt.show()
