# Import relevant packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import gurobipy as gb
from gurobipy import GRB

random.seed(101)

## Parameter Initialization
# For player A - set arbitrarily
MCa = 10        # marginal cost (€/MWh)
Q_max_a = 40    # maximum power output/input (MW)
E_max_a = 100   # maximum battery level (MWh)

# For player B - set arbitrarily
MCb = 5
Q_max_b = 60
E_max_b = 300

# Discrete bid values
N = 10  # number of discrete values the players can bid
Q_a = [Q_max_a * i / N for i in range(1,N+1)]   # list of available bids for player A
Q_b = [Q_max_b * i / N for i in range(1,N+1)]   # list of available bids for player B

# Battery parameters - set arbitrarily
eta = 0.9               # efficiency of charging
alpha_batt = 0.5        # percentage of initial battery level
tolerance_end = 0.01    # tolerance percentage between the initial and final battery level

# Time periods
T = 24
TIME = range(T)

# RES and Demand Data
RES = np.array([random.random() * (Q_max_a + Q_max_b) * 1.5 for _ in TIME])   # Power supply from renewable energy
Demand = np.array([random.random() * (Q_max_a + Q_max_b) * 2 for _ in TIME])  # Demand from the grid

Residual = RES - Demand
sign_R = np.array([0 if r <= 0 else 1 for r in Residual])  # binary parameter: 1 if residual production, 0 is residual demand

# Demand piecewise curve
P_D = np.array([10e5, 70, 60, 40, 20, 10, 0])       # price levels for demand curve
D = len(P_D)                                        # number of price levels
Step_dem = np.array([i/(D-2) for i in range(D-1)])  # intervals breakpoints
eps = min(Q_a[0], Q_b[0])/np.max(Demand)/2          # small value bigger than 0, chosen lower than the minimum value the players can bid so as not to influence the decision-making
Step_dem[0] = eps

# Big M parameter
M = [max(Demand[t], RES[t], max(Q_a), max(Q_b)) for t in range(T)]
