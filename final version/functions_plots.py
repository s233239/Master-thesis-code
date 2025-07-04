"""
Plotting utilities for electricity market data analysis.

This module provides functions to generate common visualizations such as
demand curves, renewable production, and residual demand over time.
Each function is designed to take preprocessed data and a matplotlib axes
object for flexible subplot integration.

Functions follow a consistent interface: 
    plot_x(ax, data1, data2, ...)

Currently available:
    - plot_price_demand_curve
    - plot_demand_over_time
    - plot_renewable_over_time
    - plot_residual_over_time
"""

import pandas as pd
import numpy as np


## Price Demand Curve
def plot_price_demand_curve(ax, Demand_price, Demand_volume):
    """
    Plot hourly price-demand step curves.

    Parameters:
        ax (matplotlib.axes.Axes): Plotting axes
        Demand_price (DataFrame Nx24): Price per demand step per hour
        Demand_volume (DataFrame Nx24): Volume per demand step per hour
    """
    temps = Demand_price.shape[1]
    for t in range(temps):
        ax.step(
            np.insert(Demand_volume[:, t], 0, 0),
            np.insert(Demand_price[:, t], 0, Demand_price[0, t]),
            where='post',
            label=f"Hour {t+1}"
        )
    ax.set_xlabel("Volume (MWh)")
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title("Price Demand Curve")
    ax.grid()


## Demand Over Time
def plot_demand_over_time(ax, Demand_volume_total):
    """
    Plot total demand per hour.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        Demand_volume_total (np.ndarray): Total hourly demand
    """
    temps = Demand_volume_total.shape[0]
    ax.plot(Demand_volume_total, color="red", marker='.')
    ax.bar(x=range(temps), height=Demand_volume_total, color='red', alpha=0.5, align='edge')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Cumulated demand (MWh)")
    ax.set_title("Demand Over Time")
    ax.grid()


## Renewable Production Over Time
def plot_renewable_over_time(ax, RES, Demand_volume_total):
    """
    Plot RES production and demand comparison.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        RES (Series or array): Renewable hourly production
        Demand_volume_total (np.ndarray): Hourly demand
    """
    temps = Demand_volume_total.shape[0]
    ax.plot(RES, color="green", marker='.')
    ax.bar(x=range(temps), height=RES, color='green', alpha=0.5, align='edge')
    ax.plot(Demand_volume_total, color='red', linestyle='--', linewidth=1, label="Total Demand")
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Renewable Production Over Time")
    ax.legend()
    ax.grid()


## Residual Demand Over Time
def plot_residual_over_time(ax, Residual):
    """
    Plot hourly residual demand.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        Residual (np.ndarray): Residual hourly demand
    """
    temps = Residual.shape[0]
    ax.plot(Residual, color="red", marker='.')
    ax.bar(x=range(temps), height=Residual, color='red', alpha=0.5, align='edge')
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Residual Demand Over Time")
    ax.grid()
