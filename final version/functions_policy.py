import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

# Adjust optimization objective accordingly 
def apply_policy_to_revenue(revenue: dict,
                            q_charge: dict,
                            q_discharge: dict,
                            residual_series,
                            policy_type: str = "none",
                            kwargs: dict = None) -> list:
    """
    Adjusts revenue based on selected policy.

    Parameters:
        revenue (dict): Base revenue from market arbitrage.
        q_charge (dict): Charging quantities (MWh)
        q_discharge (dict): Discharging quantities (MWh)
        residual_series (np.array or list of float): Market residual demand (MWh) (always positive - if 0 then it is possible there is residual production)
        policy_type (str):  One of ["none", "grid_tariff_flat", "grid_tariff_hourly", 
                                   "grid_tariff_dynamic", "capacity", "curtailment", "all"].
                            If "all" then also precise which grid_tariff policy should be applied.
                            Default is "none".
        kwargs: Policy-specific parameters:
            - tau_ch (float or np.ndarray): grid tariff for charging
            - tau_dis (float or np.ndarray): grid tariff for discharging
            - min_val (float)
            - max_val (float)
            - season (str)
            - alpha (float)
            - threshold (float)
            - base_tariff (float)
            - floor_tariff (float)
            - colocation_policy (bool)
            - ...
            # - P_max (float)
            # - payment_rate (float)
            # - q_charge_curt (np.ndarray)
            # - price_curt (float)

    Returns:
        adjust_to_revenue (list of float): Computed tariffs after applying policy.
    """
    TIME = range(len(residual_series))
    adjust_to_revenue = [0.0 for t in TIME]

    if policy_type == "none":
        return adjust_to_revenue
    
    if policy_type.startswith('grid_tariff'):
        # Computations of grid tariffs according to scenario
        if policy_type == "grid_tariff_flat":
            tau_ch = kwargs.get("tau_ch", [2.0 for t in TIME])
            tau_dis = kwargs.get("tau_dis", [2.0 for t in TIME])

            # Make the flat tariffs as hourly vectors
            if isinstance(tau_ch, float):
                tau_ch = [tau_ch for t in TIME]
            if isinstance(tau_dis, float):
                tau_dis = [tau_dis for t in TIME]

        elif policy_type == "grid_tariff_hourly":
            min_val = kwargs.get("min_val", 2.0)
            max_val = kwargs.get("max_val", 5.0)
            season = kwargs.get("season", "Winter")
            
            tau_ch = generate_hourly_tariff_vector('charging', season, min_val, max_val)
            tau_dis = generate_hourly_tariff_vector('discharging', season, min_val, max_val)

        elif policy_type == "grid_tariff_dynamic":
            alpha = kwargs.get("alpha", 0.001)
            threshold = kwargs.get("threshold", 0.0)
            base_tariff = kwargs.get("base_tariff", 5.0)
            floor_tariff = kwargs.get("floor_tariff", 2.0)

            # Dynamic grid tariffs based on residual demand trends   
            tau_ch = compute_charging_tariff_from_residual_demand(residual_series, alpha, threshold, base_tariff, floor_tariff)
            tau_dis = compute_discharging_tariff_from_residual_demand(residual_series, alpha, threshold, base_tariff, floor_tariff)

        # Get co-location policy parameter
        colocation_policy = kwargs.get("colocation_policy", False)
        if colocation_policy:
            tau_ch = [0 for t in TIME]
        
        # Now compute the adjusted tariff revenue based on the corresponding tariffs
        grid_cost = [(q_charge[t] * tau_ch[t]) + 
                     (q_discharge[t] * tau_dis[t]) for t in TIME]
        adjust_to_revenue = [adjust_to_revenue[t] - grid_cost[t] for t in TIME]

    else:
        raise ValueError("Wrong policy name.")
    # if policy_type in ["capacity", "all"]:
    #     E_max = kwargs.get("E_max", 0.0)
    #     payment_rate = kwargs.get("payment_rate", 0.0)
    #     adjust_to_revenue += E_max * payment_rate

    # if policy_type in ["curtailment", "all"]:
    #     q_charge_curt = kwargs.get("q_charge_curt", np.zeros_like(q_charge))
    #     price_curt = kwargs.get("price_curt", 0.0)
    #     adjust_to_revenue += np.sum(q_charge_curt) * price_curt

    return adjust_to_revenue


def generate_hourly_tariff_vector(mode='discharging', season='Summer', min_val=2.0, max_val=5.0):
    """
    Generate an hourly tariff vector for charging or discharging in summer or winter,
    scaled between min_val and max_val.

    Parameters:
        mode (str): 'charging' or 'discharging'. Determines pricing direction.
        season (str): 'summer' or 'winter'. Affects time-of-day shape.
        min_val (float): Minimum tariff value.
        max_val (float): Maximum tariff value.

    Returns:
        tariff_vector (list of float): Length-24 vector of hourly tariffs.
    """
    if mode not in ['charging', 'discharging']:
        raise ValueError("mode must be either 'charging' or 'discharging'")
    if season not in ['Summer', 'Winter','LowLoad']:
        raise ValueError("season must be either Summer, Winter or LowLoad")

    if season in ['Summer', 'LowLoad']:
        # High demand during midday -> low tariffs for discharging
        base_pattern = [
            1.0, 1.0, 1.0, 0.93, 0.93, 0.83,   # 00:00–05:00
            0.5, 0.33,                         # 06:00–07:00
            0.17, 0.0, 0.0, 0.0, 0.0, 0.0,     # 08:00–13:00
            0.07, 0.17, 0.27, 0.33,            # 14:00–17:00
            0.5, 0.67, 0.77,                   # 18:00–20:00
            0.93, 1.0, 1.0                     # 21:00–23:00
        ]
    else:  # 'Winter'
        # Winter peaks in morning and evening 
        base_pattern = [
            0.9, 0.85, 0.8, 0.75, 0.6, 0.4,     # 00:00–05:00
            0.2, 0.0, 0.0, 0.1,                 # 06:00–09:00 (peak)
            0.3, 0.4, 0.5, 0.55, 0.5,           # 10:00–14:00 (midday lull)
            0.3, 0.1, 0.0, 0.1,                 # 15:00–18:00 (peak)
            0.3, 0.5, 0.7, 0.85, 1.0            # 19:00–23:00
        ]

    # Adjust direction: discharging gets lower prices in busy hours → pattern as-is
    # Charging gets higher prices in busy hours → invert
    pattern = base_pattern if mode == 'discharging' else [1 - x for x in base_pattern]

    # Scale to min/max
    tariff_vector = [min_val + (max_val - min_val) * x for x in pattern]

    return tariff_vector


def compute_charging_tariff_from_residual_demand(residual_series, alpha=0.02, threshold=1000, base_tariff=2.0, floor_tariff=5.0):
    """
    Compute charging tariffs based on residual demand, decreasing tariffs when residual production is high.

    Parameters:
        residual_series (np.array or list of float):
            Time series of residual demand values (total demand minus RES).
        alpha (float, optional):
            Sensitivity coefficient; determines how fast the tariff decreases above the threshold.
        threshold (float, optional):
            Residual demand level above which tariffs start increasing.
        base_tariff (float, optional):
            Default tariff applied when residual demand is below threshold.
        floor_tariff (float, optional):
            Maximum tariff allowed to prevent negative or too-low tariffs.

    Returns:
        tariff (np.array of float):
            Array of discharging tariffs for each time step.
    """
    TIME = range(len(residual_series))

    residual_production = np.where(residual_series < 0, -residual_series, 0)

    tariff = [base_tariff * (1 - alpha * max(0, residual_production[t] - threshold)) for t in TIME]
    tariff = [max(tariff[t], floor_tariff) for t in TIME]

    return tariff


def compute_discharging_tariff_from_residual_demand(residual_series, alpha=0.05, threshold=1000, base_tariff=5, floor_tariff=0.5):
    """
    Compute discharging tariffs based on residual demand, decreasing tariffs when residual demand is high.

    Parameters:
        residual_series (np.array or list of float):
            Time series of residual demand values (total demand minus RES).
        alpha (float, optional):
            Sensitivity coefficient; determines how fast the tariff decreases above the threshold.
        threshold (float, optional):
            Residual demand level above which tariffs start decreasing.
        base_tariff (float, optional):
            Default tariff applied when residual demand is below threshold.
        floor_tariff (float, optional):
            Minimum tariff allowed to prevent negative or too-low tariffs.

    Returns:
        tariff (np.array of float):
            Array of discharging tariffs for each time step.
    """
    TIME = range(len(residual_series))
    
    tariff = [base_tariff * (1 - alpha * max(0, residual_series[t] - threshold)) for t in TIME]
    tariff = [max(tariff[t], floor_tariff) for t in TIME]

    return tariff



