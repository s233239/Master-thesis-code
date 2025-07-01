"""
Functions:
reduce_demand_curves, plot_reduction

Main:
Builds the input (demand_price, demand_volume) the model function needs to represent the spot market demand data.
Inputs have been reduced to N steps (in the stepwise price demand curve).
Plots the before/after reduction.
"""

import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import os

from demand_data_processing import cumulative_to_marginal


def reduce_demand_curves(demand_dict:dict, N:int):
    """
    Reduces stepwise hourly demand curves to N representative steps using weighted k-medoids clustering.

    The function performs the following:
    - Separates the step at price = 4000 and keeps it as a fixed, unclustered step (if present).
    - For all other steps, uses price as clustering variable.
    - Volumes are used as weights by repeating each price proportionally to its volume (rounded).
    - Applies k-medoids clustering to group price steps into N-1 clusters (excluding the fixed 4000 step).
    - Aggregates volumes within each cluster.
    - Sorts the resulting steps in descending order of price.

    Parameters:
        demand_dict (dict):
            Dictionary mapping each hour (int 0-23) to a DataFrame with columns ['Price', 'Volume'].
            Each DataFrame should represent the stepwise demand curve for one hour.

        N (int):
            Total number of steps desired in the reduced curve (including the fixed 4000-price step, if present).
            If the 4000-price step is not present, N clusters will be used.

    Returns:
        (demand_price, demand_volume) (pd.DataFrame, pd.DataFrame):
        -   DataFrame of shape (N, 24) with representative price steps.
            Columns are hours (0-23), rows are price steps (sorted high to low).
        -   DataFrame of shape (N, 24) with corresponding cumulative volumes per price step.
            Same structure as demand_price.
    """

    demand_price = pd.DataFrame(columns=range(24))
    demand_volume = pd.DataFrame(columns=range(24))

    for hour, df in demand_dict.items():
        # Replace cumulative volumes to marginal ones in the dataframe
        adapted_df = cumulative_to_marginal(df)
        
        # Divide the demand data into 1 cluster for price 4000 / and the rest
        high_price_mask = adapted_df['Price'] == 4000   # boolean mask, same length as adapted_df
        fixed_df = adapted_df[high_price_mask]
        cluster_df = adapted_df[~high_price_mask]

        # Extract arrays
        prices = cluster_df['Price'].values
        volumes = cluster_df['Marginal'].values

        # Apply k-medoids on prices
        if len(prices) < N:
            # Not enough points to cluster; just return original - exception case but assumed to never happen
            reduced_prices = prices
            reduced_volumes = volumes
            print("Not enough data points.")

        else:
            # Repeat prices according to volume (rounded) for weighted k-medoids clustering
            weights = np.round(volumes).astype(int)
            expanded_prices = np.repeat(prices, weights).reshape(-1, 1)

            # Run k-medoids clustering algorithm
            kmedoids = KMedoids(n_clusters=N-1, random_state=0).fit(expanded_prices)
            labels = kmedoids.predict(prices.reshape(-1, 1))
            medoids = kmedoids.cluster_centers_.flatten()

            # Aggregate volumes per medoid cluster
            reduced_prices = []
            reduced_volumes = []
            for i in range(N-1):
                cluster_mask = labels == i
                cluster_volume = volumes[cluster_mask].sum()    # works because volumes are marginals
                cluster_price = medoids[i]
                reduced_prices.append(cluster_price)
                reduced_volumes.append(cluster_volume)
            
        # Add the fixed 4000-price point
        if not fixed_df.empty:
            reduced_prices.append(4000)
            reduced_volumes.append(fixed_df['Marginal'].sum())

        # Sort the clusters by price in descending order
        sorted_idx = np.argsort(reduced_prices)[::-1]
        demand_price[hour] = np.array(reduced_prices)[sorted_idx]

        # Convert marginal volumes to cumulative volumes
        marginal_sorted = np.array(reduced_volumes)[sorted_idx]
        demand_volume[hour] = np.cumsum(marginal_sorted)


    # Set index to step (0 to N-1)
    demand_price.index = range(N)
    demand_volume.index = range(N)

    return demand_price, demand_volume



def plot_reduction(demand_dict, reduced_price_df, reduced_volume_df, hours_to_plot):
    for hour in hours_to_plot:
        df = demand_dict[hour]
        prices = df['Price'].values
        volumes = df['Volume'].values

        red_prices = reduced_price_df[hour].dropna().values
        red_volumes = reduced_volume_df[hour].dropna().values

        plt.figure(figsize=(6, 4))
        plt.step(prices, volumes, where='post', label='Original', alpha=0.6)
        plt.step(red_prices, red_volumes, where='post', label='Reduced', color='r', linewidth=2)
        plt.title(f'Demand Curve - Hour {hour}')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# reduced_price_df, reduced_volume_df = reduce_demand_curves_weighted(demand_dict, N=5)
# plot_reduction(demand_dict, reduced_price_df, reduced_volume_df, hours_to_plot=[7, 13, 18])
