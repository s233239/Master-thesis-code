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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "2"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*KMeans is known to have a memory leak.*")

from demand_data_processing import cumulative_to_marginal, load_hourly_demand_curves


def reduce_demand_curves(demand_dict:dict, N:int):
    """
    Reduces stepwise hourly demand curves to N representative steps using weighted K-Means clustering algorithm.
    Returns the desired input for optimization model as 2 dataframes for Price and Volume.

    The function performs the following:
    - Separates the bids at price = 4000 and keeps it as a fixed, unclustered step (if present).
    - For all other steps, uses price as clustering variable. Volumes are used as weights.
    - Applies K-Means clustering to price steps into N-1 clusters (excluding the fixed 4000 step).
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
        adapted_df = adapted_df[adapted_df['Marginal'] > 0]  # filter out 0-volume steps
        
        # Divide the demand data into 1 cluster for price 4000 / and the rest
        high_price_mask = adapted_df['Price'] == 4000   # boolean mask, same length as adapted_df
        fixed_df = adapted_df[high_price_mask]
        cluster_df = adapted_df[~high_price_mask]

        # Extract arrays
        prices = cluster_df['Price'].to_numpy()
        volumes = cluster_df['Marginal'].to_numpy()

        # Run K-Means clustering algorithm
        if len(prices) < N:
            # Not enough points to cluster; just return original - exception case but assumed to never happen
            reduced_prices = prices
            reduced_volumes = volumes
            print("Not enough data points.")
        else:
            prices_reshaped = prices.reshape(-1, 1)

            kmeans = KMeans(n_clusters=N-1, n_init=10, random_state=0)
            kmeans.fit(prices_reshaped, sample_weight=volumes)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_.flatten()

            # Check for empty clusters / warnings for cluster count
            unique_labels = np.unique(labels)
            if len(unique_labels) < N - 1:
                print(f"Warning: Only {len(unique_labels)} clusters found instead of {N-1}")

            # Aggregate volumes per medoid cluster
            reduced_prices = []
            reduced_volumes = []
            for i in range(N-1):
                cluster_mask = labels == i
                cluster_volume = volumes[cluster_mask].sum()    # works because volumes are marginals
                cluster_price = centroids[i]
                reduced_prices.append(cluster_price)
                reduced_volumes.append(cluster_volume)
            
        # Add the fixed 4000-price cluster
        if not fixed_df.empty:
            reduced_prices.append(4000)
            reduced_volumes.append(fixed_df['Marginal'].sum())

        # Sort the clusters by price in descending order
        sorted_idx = np.argsort(reduced_prices)[::-1]
        demand_price[hour] = np.array(reduced_prices)[sorted_idx]

        # Convert back marginal volumes to cumulative volumes
        marginal_sorted = np.array(reduced_volumes)[sorted_idx]
        demand_volume[hour] = np.cumsum(marginal_sorted)


    # Set index to step (0 to N-1)
    demand_price.index = range(N)
    demand_volume.index = range(N)

    return demand_price, demand_volume


def plot_reduction(demand_dict, reduced_price_df, reduced_volume_df, hours_to_plot):

    # Create a subplot
    n_plots = len(hours_to_plot)
    cols = np.ceil(np.sqrt(n_plots)).astype(int)
    rows = np.ceil(n_plots / cols).astype(int)

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    np.array(axes).flatten()

    for hour in hours_to_plot:
        df = demand_dict[hour]
        prices = df['Price'].to_numpy()
        volumes = df['Volume'].to_numpy()

        red_prices = reduced_price_df[hour].to_numpy()
        red_volumes = reduced_volume_df[hour].to_numpy()

        plt.subplot(rows, cols, hours_to_plot.index(hour)+1)
        plt.step(np.insert(volumes, 0, 0), np.insert(prices, 0, prices[0]), label='Original', alpha=0.8)
        plt.step(np.insert(red_volumes, 0, 0), np.insert(red_prices, 0, red_prices[0]), label='Reduced', color='r')
        plt.title(f'Demand Curve - Hour {hour}')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return



# # == MAIN == 
# # Get directory path where the script is located
# script_dir = os.path.dirname(__file__)

# # Set data file paths
# dk1_summer_file = 'auction_aggregated_curves_dk1_20240710.csv'
# dk1_winter_file = 'auction_aggregated_curves_dk1_20241127.csv'
# dk1_lowload_file = 'auction_aggregated_curves_dk1_20240519.csv'
# dk2_summer_file = 'auction_aggregated_curves_dk2_20240710.csv'
# dk2_winter_file = 'auction_aggregated_curves_dk2_20241127.csv'
# dk2_lowload_file = 'auction_aggregated_curves_dk2_20240519.csv'

# # Choose scenario
# # bidding_zone  \in {'dk1', 'dk2'}
# # scenario      \in {'winter', 'summer', 'lowload'}
# csv_filename = dk2_winter_file
# label = "DK2 Winter"

# # Build relative path to CSV file in the same folder
# csv_path = os.path.join(script_dir, csv_filename)

# # Get aggregated demand curve for all hours
# price_demand_curves = load_hourly_demand_curves(csv_path)
    
# # Test the processing functions
# reduced_price_df, reduced_volume_df = reduce_demand_curves(price_demand_curves, N=20)
# plot_reduction(price_demand_curves, reduced_price_df, reduced_volume_df, hours_to_plot=[0, 7, 13, 18])
