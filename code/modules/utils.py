"""
This module contains utility functions like plotting.
"""


# Imports
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union

# Functions

def add_gaussian_noise(field: np.ndarray,
                       noise_percentage: float,
                       seed: int) -> np.ndarray:
    """
    Function to create and add noise to a given field.

    Parameters
    ----------
    field : np.ndarray
        Input field.
    noise_percentage : float
        Noise percentage to setup noise amplitude in relation to maximum value of input field.
    seed : int
        Integer to start seed.

    Returns
    -------
    noisy_field : np.ndarray
        Field with noise.
    noise : np.ndarray
        Noise sequence.
    sigma : float
        Parameter to define noise scale.

    """
    rng = np.random.default_rng(seed)
    max_amplitude = np.max(np.abs(field))
    sigma = (noise_percentage / 100.0) * max_amplitude
    noise = rng.normal(loc=0.0, scale=sigma, size=field.shape)
    noisy_field = field + noise

    return noisy_field


def plot_field_contourf(xcoords: np.ndarray, 
                        ycoords: np.ndarray,
                        field: np.ndarray,
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        unit: str,
                        figname: (str | Path),
                        show: bool = False) -> None:
    """
    Function to plot field data.

    Parameters
    ----------
    xcoords : 2d Array
        2D array containing the coordinates in the X axis, Cartesian System.
    ycoords : TYPE
        DESCRIPTION.
    field : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    unit : TYPE
        DESCRIPTION.
    figname : TYPE
        DESCRIPTION.
    mode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig = plt.figure()
    plt.contourf(xcoords, ycoords, field)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_aspect("equal")
    plt.colorbar(label=unit)
    fig.savefig(figname, dpi=300, bbox_inches="tight")

    # Handling Display vs. Memory
    if show:
        plt.show()
    else:
        plt.close(fig)

