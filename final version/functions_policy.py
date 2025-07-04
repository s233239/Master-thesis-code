
import numpy as np
from ctypes import Union

# Adjust optimization objective accordingly 
def apply_policy_to_revenue(revenue: float,
                            q_charge: np.ndarray,
                            q_discharge: np.ndarray,
                            price: np.ndarray,
                            policy_type: str = "none",
                            kwargs: dict = None) -> float:
    """
    Adjusts revenue based on selected policy.

    Parameters:
        revenue (float): Base revenue from market arbitrage.
        q_charge (np.ndarray): Charging quantities (MWh)
        q_discharge (np.ndarray): Discharging quantities (MWh)
        price (np.ndarray): Market prices (€/MWh)
        policy_type (str):  One of ["none", "grid_tariff_flat", "grid_tariff_hourly", 
                                   "grid_tariff_dynamic", "capacity", "curtailment", "all"]
                            If "all" then also precise which grid_tariff policy should be applied.
        kwargs: Policy-specific parameters:
            - tau_ch (float or np.ndarray): grid tariff for charging
            - tau_dis (float or np.ndarray): grid tariff for discharging
            - alpha (float)
            - P_max (float)
            - payment_rate (float)
            - q_charge_curt (np.ndarray)
            - price_curt (float)
            - threshold (float)
            - base_tariff (float)

    Returns:
        adjusted_revenue (float): Modified revenue after applying policy.
    """
    
    adjusted_revenue = revenue

    if policy_type == "none":
        return adjusted_revenue

    if policy_type in ["grid_tariff_flat", "grid_tariff_hourly"]:
        tau_ch = kwargs.get("tau_ch", 0.0)
        tau_dis = kwargs.get("tau_dis", 0.0)
        grid_cost = np.sum(q_charge * tau_ch) + np.sum(q_discharge * tau_dis)
        adjusted_revenue -= grid_cost

    elif policy_type in ["grid_tariff_dynamic"]:
        alpha = kwargs.get("alpha", 0.01)
        threshold = kwargs.get("threshold", None)
        base_tariff = kwargs.get("base_tariff", 2.0)
        floor_tariff = kwargs.get("floor_tariff", 0.0)

        # Dynamic grid tariffs for charging: useless for our model since storage charge when price is zero
        # if threshold is None:
        #     tau_ch = alpha * price
        # else:
        #     # Price-threshold tariff will either be computed with a flat tariff or a market-based tariff (depending on which parameters are defined)
        #     tau_ch = np.where(price < threshold, 0.0, base_tariff + alpha*(price-threshold))
        tau_ch = kwargs.get("tau_ch", 0.0)

        # Dynamic grid tariffs for discharging (with a decreasing affine function)   
        tau_dis = compute_discharging_tariff_from_price(price, alpha, threshold, base_tariff, floor_tariff)

        grid_cost = np.sum(q_charge * tau_ch) + np.sum(q_discharge * tau_dis)
        adjusted_revenue -= grid_cost

    if policy_type in ["capacity", "all"]:
        E_max = kwargs.get("E_max", 0.0)
        payment_rate = kwargs.get("payment_rate", 0.0)
        adjusted_revenue += E_max * payment_rate

    if policy_type in ["curtailment", "all"]:
        q_charge_curt = kwargs.get("q_charge_curt", np.zeros_like(q_charge))
        price_curt = kwargs.get("price_curt", 0.0)
        adjusted_revenue += np.sum(q_charge_curt) * price_curt

    return adjusted_revenue


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
    if threshold is None:
        if alpha > 0:
            # tariff = base_tariff - alpha * price
            tariff = base_tariff * (1 - alpha * price_series)
        else:
            tariff = floor_tariff            
    else:
        if alpha > 0:
            # tariff = base_tariff - alpha * (price - threshold)
            tariff = np.where(price_series < threshold, base_tariff, base_tariff * (1 - alpha * (price_series - threshold)))
        else:
            tariff = np.where(price_series < threshold, base_tariff, floor_tariff)

    tariff = np.where(tariff > floor_tariff, tariff, floor_tariff)

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

