
import numpy as np
import gurobipy as gp

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
            - alpha (float)
            - threshold (float)
            - base_tariff (float)
            - floor_tariff (float)
            - P_max (float)
            - payment_rate (float)
            - q_charge_curt (np.ndarray)
            - price_curt (float)

    Returns:
        adjust_to_revenue (list of float): Computed tariffs after applying policy.
    """
    TIME = range(len(residual_series))
    adjust_to_revenue = [0.0 for t in TIME]

    if policy_type == "none":
        return adjust_to_revenue

    if policy_type in ["grid_tariff_flat", "grid_tariff_hourly"]:
        tau_ch = kwargs.get("tau_ch", [2.0 for t in TIME])
        tau_dis = kwargs.get("tau_dis", [2.0 for t in TIME])
        if isinstance(tau_ch, float):
            tau_ch = [tau_ch for t in TIME]
        if isinstance(tau_dis, float):
            tau_dis = [tau_dis for t in TIME]

        grid_cost = [(q_charge[t] * tau_ch[t]) + 
                    (q_discharge[t] * tau_dis[t]) for t in TIME]
        adjust_to_revenue = [adjust_to_revenue[t] - grid_cost[t] for t in TIME]

    elif policy_type in ["grid_tariff_dynamic"]:
        alpha = kwargs.get("alpha", 0.01)
        threshold = kwargs.get("threshold", None)
        base_tariff = kwargs.get("base_tariff", 5.0)
        floor_tariff = kwargs.get("floor_tariff", 0.0)

        # Dynamic grid tariffs for charging: useless for our model since storage usually charge when price is zero
        tau_ch = kwargs.get("tau_ch", [2.0 for t in TIME])
        if isinstance(tau_ch, float):
            tau_ch = [tau_ch for t in TIME]

        # Dynamic grid tariffs for discharging (decreasing when residual demand increases)   
        tau_dis = compute_discharging_tariff_from_residual_demand(residual_series, alpha, threshold, base_tariff, floor_tariff)

        grid_cost = [(q_charge[t] * tau_ch[t]) + 
                    (q_discharge[t] * tau_dis[t]) for t in TIME]
        adjust_to_revenue = [adjust_to_revenue[t] - grid_cost[t] for t in TIME]

    if policy_type in ["capacity", "all"]:
        E_max = kwargs.get("E_max", 0.0)
        payment_rate = kwargs.get("payment_rate", 0.0)
        adjust_to_revenue += E_max * payment_rate

    if policy_type in ["curtailment", "all"]:
        q_charge_curt = kwargs.get("q_charge_curt", np.zeros_like(q_charge))
        price_curt = kwargs.get("price_curt", 0.0)
        adjust_to_revenue += np.sum(q_charge_curt) * price_curt

    return adjust_to_revenue


def compute_discharging_tariff_from_residual_demand(residual_series, alpha=0.05, threshold=1000, base_tariff=5, floor_tariff=0.5):
    """
    Compute discharging tariffs based on residual demand, decreasing tariffs when residual demand is high.

    Parameters:
        residual_series (np.array or list of float):
            Time series of residual demand values (e.g., demand minus RES).
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

    if threshold is None:
        threshold = 0.0
    
    tariff = [base_tariff * (1 - alpha * max(0, residual_series[t] - threshold)) for t in TIME]
    tariff = [max(tariff[t], floor_tariff) for t in TIME]

    return tariff


# Deprecated because of non-linear computations
def compute_discharging_tariff_from_price(price_series, alpha=0.01, threshold=10, base_tariff=5, floor_tariff=0):
    """
    Compute dynamic discharging grid tariff based on market price.

    Parameters:
        price_series (list or array): Market price time series (€/MWh)
        alpha (float): Sensitivity of the tariff to price above threshold.
        threshold (float): Price (€/MWh) above which tariff starts decreasing.
        base_tariff (float): Base discharging grid tariff (€/MWh).
        floor_tariff (float): Minimum allowable grid tariff (€/MWh).

    Returns:
        discharging_tariff (list): Hourly grid tariffs for discharging (€/MWh)
    """
    TIME = range(len(price_series))
    if threshold is None:
        # tariff = base_tariff - alpha * price
        tariff = [base_tariff * (1 - alpha * price_series[t]) for t in TIME]       
    else:
        raise ValueError("Dynamic tariffs have not been modelled to include a price threshold from which tariffs will decrease [ongoing]")
        # if alpha > 0:
        #     # tariff = base_tariff - alpha * (price - threshold)
        #     tariff = np.where(price_series < threshold, base_tariff, base_tariff * (1 - alpha * (price_series - threshold)))
        # else:
        #     tariff = np.where(price_series < threshold, base_tariff, floor_tariff)

    for t in TIME:
        tariff[t] = gp.max_(tariff[t], floor_tariff)

    return tariff


# # Aggregator Function for All Policies
# def apply_storage_policies(q_charge: np.ndarray, q_discharge: np.ndarray,
#                            tau_ch: Union[float, np.ndarray], tau_dis: Union[float, np.ndarray],
#                            P_max: float, payment_rate: float,
#                            q_charge_curt: np.ndarray, price_curt: float) -> float:
#     """
#     Aggregates the economic effects of all three policies on storage.

#     Returns:
#         total_policy_value (float): Combined revenue/cost effect from all policies.
#     """
#     tariff_cost = grid_tariff_cost(q_charge, q_discharge, tau_ch, tau_dis)
#     cap_payment = capacity_payment(P_max, payment_rate)
#     curt_revenue = curtailment_compensation(q_charge_curt, price_curt)
#     return -tariff_cost + cap_payment + curt_revenue


# # Grid Tariff Policy
# def grid_tariff_cost(q_charge: np.ndarray, q_discharge: np.ndarray,
#                      tau_ch: Union[float, np.ndarray], tau_dis: Union[float, np.ndarray]) -> float:
#     """
#     Computes the total grid tariff cost applied to charging and discharging.

#     Parameters:
#         q_charge (np.ndarray): Charging quantities (MWh per hour)
#         q_discharge (np.ndarray): Discharging quantities (MWh per hour)
#         tau_ch (float or np.ndarray): Charging tariff (€/MWh)
#         tau_dis (float or np.ndarray): Discharging tariff (€/MWh)

#     Returns:
#         grid_cost (float): Total cost from applying tariffs on storage operation.
#     """
#     return np.sum(q_charge * tau_ch) + np.sum(q_discharge * tau_dis)


# # Capacity Payment Policy
# def capacity_payment(P_max: float, payment_rate: float) -> float:
#     """
#     Computes the total capacity remuneration for a given max power.

#     Parameters:
#         P_max (float): Maximum discharge capacity (MW)
#         payment_rate (float): Payment rate (€/MW capacity)

#     Returns:
#         payment (float): Capacity payment received.
#     """
#     return P_max * payment_rate


# # Curtailment Compensation Policy
# def curtailment_compensation(q_charge_curt: np.ndarray, price_curt: float) -> float:
#     """
#     Computes compensation for charging from curtailed renewable energy.

#     Parameters:
#         q_charge_curt (np.ndarray): Charging from curtailed energy (MWh per hour)
#         price_curt (float): Compensation rate (€/MWh charged from curtailment)

#     Returns:
#         compensation (float): Total compensation received for curtailment absorption.
#     """
#     return np.sum(q_charge_curt) * price_curt

