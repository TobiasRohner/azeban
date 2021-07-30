#!/usr/bin/env python3

import sys
import pickle
import lmmr
import numpy as np


class ConvergencePlot(lmmr.io.convergence_plots.SplitConvergencePlot):

    def __init__(self):
        super().__init__()
        self.colors = dict()

    def error_style(self, method):
        if not method in self.colors:
            self.colors[method] = 'C{}'.format(len(self.colors))
        return {'color': self.colors[method], 'label':method}

    def rate_style(self, method):
        if not method in self.colors:
            self.colors[method] = 'C{}'.format(len(self.colors))
        return {'color': self.colors[method]}


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
    plot = ConvergencePlot()
    for method in data:
        dat = data[method]
        Ns = [t[0] for t in dat]
        dx = 1 / np.asarray(Ns)
        errs = [t[1] for t in dat]
        rates = lmmr.io.convergence_plots.compute_rates(dx, errs)
        plot.add(dx, errs, rates, method)
    plot.ax1.legend()
    plot.finalize('L2 Error of Taylor Vortex in 2D', 'L2 error')
    plot.save(sys.argv[2])
