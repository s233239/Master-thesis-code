"""
Functions:
- plot_market_price_summary
- plot_stacked_welfare
- plot_curtail_unmet_groupedbars, plot_curtailment_and_unmet_subplots
- plot_revenue_per_mwh
- plot_total_discharge
"""


# Import relevant packages
import pandas as pd                   # DataFrames
import matplotlib.pyplot as plt       # Plotting
import numpy as np                    # Numerical operations (similar to Julia base)

import warnings
warnings.filterwarnings("ignore", message=".*All values for SymLogScale are below linthresh.*")


def plot_market_price_summary(results, season="Winter", player_counts=[0,1,2,4,6,8], selected_hours=[8, 12, 18], save=False):
    avg_prices, peak_prices, weighted_avg_prices = [], [], []
    hour_prices = {h: [] for h in selected_hours}

    for n in player_counts:
        output = results[season][n]
        price = output[0][3]

        avg_prices.append(np.mean(price))
        peak_prices.append(np.max(price))
        for h in selected_hours:
            hour_prices[h].append(price[h])
    
        if not (type(n) is int and n == 0):
            discharge = [sum(output[p][1][t] for p in output) for t in range(len(price))]
            weighted_avg_prices.append(np.average(price, weights=discharge))
    
    if player_counts[0] == 0:
        player_counts_without_zero = player_counts[1:]
    else:
        player_counts_without_zero = player_counts

    plt.figure(figsize=(10, 6))
    plt.plot(player_counts, avg_prices, 'o-', label="Average Price")
    plt.plot(player_counts, peak_prices, 'o-', label="Peak Price")
    plt.plot(player_counts_without_zero, weighted_avg_prices, 'o-', label="Weighted Avg Price")

    for i in range(len(player_counts)):
        plt.text(player_counts[i], avg_prices[i] * 1.1, f"{avg_prices[i]:.2f}", ha='center', va='bottom', fontsize=9)
        plt.text(player_counts[i], peak_prices[i] * 1.03, f"{peak_prices[i]:.2f}", ha='center', va='bottom', fontsize=9)
        
    for i in range(len(player_counts_without_zero)):        
        plt.text(player_counts_without_zero[i], weighted_avg_prices[i] * 1.05, f"{weighted_avg_prices[i]:.2f}", ha='center', va='bottom', fontsize=9)

    for h in selected_hours:
        plt.plot(player_counts, hour_prices[h], 'o--', label=f"Price at {h}:00")

    plt.xlabel("Number of Players")
    plt.ylabel("Market Price [€/MWh]")
    plt.ylim(top=max(peak_prices)*1.1)
    plt.title(f"Market Price Summary - {season}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"figs_competition/market_price_summary_{season}.pdf")
        results_df = pd.DataFrame({
            "average_prices": avg_prices,
            "weighted_average_prices": np.insert(weighted_avg_prices, 0, 0),
            "peak_prices": peak_prices
        })
        results_df.to_csv(f"csv_competition\{season}-market_summary.csv")
    plt.show()

def plot_stacked_welfare(results, season="Winter", player_counts=[0,1,2,4,6,8], save=False):
    cs_list, ps_list = [], []
    for n in player_counts:
        output = results[season][n]
        cs_list.append(sum(output[0][7])/1e3)
        ps_list.append(sum(output[0][8])/1e3)

    ind = np.arange(start=1, stop=1+len(player_counts))
    width = 0.6

    plt.figure(figsize=(8, 5))
    plt.bar(ind, cs_list, width, label='Consumer Surplus', color='skyblue')
    plt.bar(ind, ps_list, width, bottom=cs_list, label='Producer Surplus', color='orange')
    plt.xticks(ind, player_counts)
    plt.xlabel("Number of Players")
    plt.ylabel("Monetary Value [thousands €]")
    plt.title(f"Evolution of Market Welfare under Competition - {season}")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"figs_competition/stacked_surplus_{season}.pdf")
        results_df = pd.DataFrame({
            "consumer_surplus": cs_list,
            "producer_surplus": ps_list
        })
        results_df.to_csv(f"csv_competition\{season}-welfare_metrics.csv")
    plt.show()

def plot_curtail_unmet_groupedbars(results, season="Winter", player_counts=[0,1,2,4,6,8], save=False):
    curtailed, unmet = [], []
    for n in player_counts:
        output = results[season][n]
        curtailed.append(np.mean(output[0][10]))
        unmet.append(np.mean(output[0][9]))

    x = np.arange(start=1, stop=1+len(player_counts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bar1 = ax.bar(x - width/2, curtailed, width, label='Curtailed Energy', color='salmon')
    bar2 = ax.bar(x + width/2, unmet, width, label='Unmet Demand', color='steelblue')

    ax.set_xlabel("Number of Players")
    ax.set_ylabel("Energy [MWh]")
    ax.set_title(f"Curtailed and Unmet Energy - {season}")
    ax.set_xticks(x)
    ax.set_xticklabels(player_counts)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add text annotations
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            offset = max(0.01 * max(curtailed + unmet), 0.05)  # adaptive offset
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.1f}",
                ha='center', va='bottom',
                fontsize=9
            )
    ax.set_ylim(top=max(curtailed)*1.05, bottom=min(unmet)*0.8)

    plt.tight_layout()
    if save:
        plt.savefig(f"figs_competition/curtailment_unmet_grouped_{season}.pdf")
    plt.show()

def plot_curtailment_and_unmet_subplots(results, season="Winter", player_counts=[0,1,2,4,6,8], mode="Total", save=False):
    curtailed, unmet = [], []
    curtailed_total, curtailed_average, unmet_total, unmet_average = [], [], [], []
    for n in player_counts:
        output = results[season][n]

        curtailed_total.append(sum(output[0][10]))
        unmet_total.append(sum(output[0][9]))
        curtailed_average.append(np.mean(output[0][10]))
        unmet_average.append(np.mean(output[0][9]))

    if mode == "Total":
        curtailed = curtailed_total
        unmet = unmet_total
    else:
        curtailed = curtailed_average
        unmet = unmet_average

    x = np.arange(len(player_counts))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- Curtailed Energy ---
    bars1 = ax1.bar(x, curtailed, color='salmon')
    ax1.set_title(f"{mode} Curtailed Energy")
    ax1.set_xlabel("Number of Players")
    ax1.set_ylabel("Energy [MWh]")
    ax1.set_ylim(top=max(curtailed)*1.05, bottom=min(curtailed)*0.75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(player_counts)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        height = bar.get_height()
        offset = max(0.01 * max(curtailed), 0.05)
        ax1.text(bar.get_x() + bar.get_width()/2, height + offset,
                 f"{height:.1f}", ha='center', va='bottom', fontsize=9)

    # --- Unmet Demand ---
    bars2 = ax2.bar(x, unmet, color='steelblue')
    ax2.set_title(f"{mode} Unmet Demand")
    ax2.set_xlabel("Number of Players")
    ax2.set_xticks(x)
    ax2.set_xticklabels(player_counts)
    ax2.set_ylim(top=max(unmet)*1.1, bottom=min(unmet)*0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        height = bar.get_height()
        offset = max(0.01 * max(unmet), 0.05)
        ax2.text(bar.get_x() + bar.get_width()/2, height + offset,
                 f"{height:.1f}", ha='center', va='bottom', fontsize=9)

    fig.suptitle(f"Evolution of Market Flexibility Metrics under Competition - {season}", fontsize=14)
    # plt.tight_layout()
    
    if save:
        plt.savefig(f"figs_competition/curtailment_unmet_{mode}_{season}.pdf")
        results_df = pd.DataFrame({
            "curtailement-average": curtailed_average,
            "unmet_demand-average": unmet_average,
            "curtailement-total": curtailed_total,
            "unmet_demand-total": unmet_total           
        })
        results_df.to_csv(f"csv_competition\{season}-flexibility_metrics.csv")
    plt.show()

def plot_revenue_per_mwh(results, season="Winter", player_counts=[1, 2, 4, 6, 8], save=False):
    rev_per_mwh = []
    weighted_avg_rev_per_mwh = []
    rev_per_capacity = []

    capacity = {
        "Winter": 6330.0,
        "Summer": 11520.0,
        "LowLoad": 9140.0
    }

    for n in player_counts:
        output = results[season][n]

        total_rev, total_dis = 0, 0
        for p in output:
            revenue = sum(output[p][4])
            discharged = sum(output[p][1])
            total_rev += revenue
            total_dis += discharged
        avg = total_rev / total_dis if total_dis > 0 else 0
        rev_per_mwh.append(avg)
        rev_per_capacity.append(total_rev / capacity[season])

        revenue, discharged = [], []
        for t in range(24):
            revenue.append(sum(output[p][4][t] for p in output))
            discharged.append(sum(output[p][1][t] for p in output))
        weighted_avg_rev_per_mwh.append(np.average(revenue, weights=discharged))


    plt.figure(figsize=(8, 5))
    plt.plot(player_counts, rev_per_mwh, 'o-', color='darkgreen', label='Revenue per MWh')
    plt.plot(player_counts, rev_per_capacity, 'o-', label='Revenue per Capacity')
    # plt.plot(player_counts, weighted_avg_rev_per_mwh, 'o-', label='Weighted Average Revenue per MWh')

    for i, val in enumerate(rev_per_mwh):
        plt.text(player_counts[i], val*1.03, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    for i, val in enumerate(rev_per_capacity):
        plt.text(player_counts[i], val*1.1, f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    # for i, val in enumerate(weighted_avg_rev_per_mwh):
    #     plt.text(player_counts[i], val*1.01, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    
    bottom, top = plt.ylim()
    plt.ylim(top=1.15*top)

    plt.xlabel("Number of Players")
    plt.ylabel("Revenue [€/MWh]")
    plt.title(f"Average Revenue Summary - {season}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"figs_competition/revenue_summary_{season}.pdf")
    plt.show()

def plot_total_discharge(results, season="Winter", player_counts=[1, 2, 4, 6, 8], save=False):
    total_discharge = []

    for n in player_counts:
        output = results[season][n]
        total_dis = sum(sum(output[p][1]) for p in output)
        total_discharge.append(total_dis)

    plt.figure(figsize=(8, 5))
    plt.plot(player_counts, total_discharge, 'o-', color='slateblue')
    plt.xlabel("Number of Players")
    plt.ylabel("Total Energy Discharged [MWh]")
    plt.title(f"Total Discharge - {season}")
    plt.grid(True)

    # Dynamic y-axis and text settings
    sorted_vals = sorted(total_discharge)
    if len(sorted_vals) >= 2 and abs(sorted_vals[-1] - sorted_vals[0]) < 1:
        # Minor differences in values → Using linear y-scale
        plt.ylim(bottom=sorted_vals[0]-1, top=sorted_vals[-1]+1)
        for i, val in enumerate(total_discharge):
            plt.text(player_counts[i], val+0.1, f"{val:.1f}", ha='center', va='bottom', fontsize=9)
    # elif len(sorted_vals) >= 2 and sorted_vals[-1] > 1.5 * sorted_vals[-2]:
    #     # Outlier value → Using symlog y-scale with linthresh = {linthresh:.1f}
    #     linthresh = sorted_vals[-2] * 1.2  # Small buffer above 2nd-largest
    #     plt.yscale('symlog', linthresh=linthresh)
    else:
        # Usual case → Using linear y-scale
        plt.ylim(top=sorted_vals[-1]*1.05)
        for i, val in enumerate(total_discharge):
            plt.text(player_counts[i], val * 1.02, f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    
    plt.tight_layout()

    if save:
        plt.savefig(f"figs_competition/total_discharge_{season}.pdf")
        results_df = pd.DataFrame({
            "total_discharge": total_discharge        
        })
        results_df.to_csv(f"csv_competition\{season}-storage_discharge.csv")
    plt.show()
