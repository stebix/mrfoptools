"""
Tooling for interactive visualization of the optimization variables.

@Author: Jannik Stebani 2025
"""
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as wgt
import jax
import jax.numpy as jnp
import numpy as np


def create_slider_callback(
    slider: wgt.IntSlider,
    line: matplotlib.lines.Line2D,
    data: np.ndarray,
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes
) -> callable:
    
    def callback(change):
        index = change['new']
        xdata = line.get_xdata()
        ydata = jnp.rad2deg(data[index])
        line.set_data(xdata, ydata)
        # ax.draw_artist(line)
        # ax.plot(xdata, ydata)
        fig.canvas.draw_idle()
        return None
    
    return callback
        

def create_sig_slider_callback(
    lines: list[matplotlib.lines.Line2D],
    data: np.ndarray,
    fig: matplotlib.figure.Figure,
    signals: jax.Array | np.ndarray
) -> callable:
    
    def callback(change):
        index = change['new']
        xdata = lines[0].get_xdata()
        signals_i = signals[index]
        for i, line in enumerate(lines):    
            ydata = signals_i[i, :]
            line.set_data(xdata, ydata.imag)
            
        fig.canvas.draw_idle()
        return None
    
    return callback