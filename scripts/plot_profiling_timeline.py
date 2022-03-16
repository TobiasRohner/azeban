#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def read_intervals(fname, from_t=None, to_t=None):
    with open(fname, 'r') as f:
        lines = f.readlines()
        data_host = lines[0].split()
        data_device = []
        if len(lines) > 1:
            data_device = lines[1].split()
    intervals_host = zip(data_host[-3::-3], data_host[-2::-3], data_host[-1::-3])
    intervals_host = [(t[0], float(t[1]), float(t[2])) for t in intervals_host if (from_t is None or float(t[1]) > from_t) and (to_t is None or float(t[1]) < to_t)]
    intervals_device = zip(data_device[-3::-3], data_device[-2::-3], data_device[-1::-3])
    intervals_device = [(t[0], float(t[1]), float(t[2])) for t in intervals_device if (from_t is None or float(t[1]) > from_t) and (to_t is None or float(t[1]) < to_t)]
    return intervals_host, intervals_device


def remove_overlap(intervals, eps=0.001):
    result = []
    for interval in intervals:
        inserted = False
        for i in range(len(result)):
            last_interval = result[i][-1]
            end_time = last_interval[1] + last_interval[2]
            if interval[1] >= end_time - eps:
                result[i].append(interval)
                inserted = True
                break
        if not inserted:
            result.append([interval])
    return result


def plot_timeline(ax, timeline, offset, color='b'):
    xmin = 1e20
    xmax = 0
    ymin = offset - 1
    ymax = ymin + len(timeline) + 1
    for i, level in enumerate(timeline):
        for interval in level:
            start = interval[1]
            duration = interval[2]
            ax.add_patch(patches.Rectangle((start, i+offset-0.5+0.1), duration, 0.8, facecolor=color, edgecolor='k'))
            ax.text(start+duration/2, i+offset, interval[0], horizontalalignment='center', verticalalignment='center', rotation='vertical')
            xmin = min(xmin, start)
            xmax = max(xmax, start + duration)
    return (xmin, xmax), (ymin, ymax)



if __name__ == '__main__':
    profiling_files = sys.argv[1:]
    fig, ax = plt.subplots()
    for fname in profiling_files:
        data_host, data_device = read_intervals(fname, 0, 100000)
        data_host = remove_overlap(data_host)
        data_device = remove_overlap(data_device)
        xlim_host, ylim_host = plot_timeline(ax, data_host, 1, 'b')
        xlim_device, ylim_device = plot_timeline(ax, data_device, len(data_host)+1, 'orange')
        xlim = (min(xlim_host[0], xlim_device[0]), max(xlim_host[1], xlim_device[1]))
        ylim = (min(ylim_host[0], ylim_device[0]), max(ylim_host[1], ylim_device[1]))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.show() 
