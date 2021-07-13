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
        return {'color': self.colors[method]}

    def rate_style(self, method):
        if not method in self.colors:
            self.colors[method] = 'C{}'.format(len(self.colors))
        return {'color': self.colors[method]}


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        data = pickle.load(f)
    Ns = [t[0] for t in data]
    dx = 1 / np.asarray(Ns)
    errs = [t[1] for t in data]
    rates = lmmr.io.convergence_plots.compute_rates(dx, errs)
    """
    dx_plot = [dx[0]]
    errs_plot = [errs[0]]
    rates_plot = [rates[0]]
    for i in range(1, len(dx)-1):
        dx_plot += [dx[i], dx[i]]
        errs_plot += [errs[i], errs[i]]
        rates_plot += [rates[i-1], rates[i]]
    dx_plot.append(dx[-1])
    errs_plot.append(errs[-1])
    rates_plot.append(rates[-1])
    """
    plot = ConvergencePlot()
    plot.add(dx, errs, rates, 'spectral')
    plot.finalize('L2 Error of Taylor Vortex in 2D', 'L2 error')
    plot.save(sys.argv[2])
