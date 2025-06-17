import random

# Set plot parameters
""" sb.set_style('ticks')
size_pp = 15
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': size_pp,
        } """
linestyles = ["-","--","-.",":"]


# Define problem auxiliary parameters
random.seed(101)
n_load = 10 #number of loads
n_hour_load = 1

# Define auxiliary functions
def generate_demand_function(n, RANGE):
    percentages = [random.random() for _ in range(n)]  # Creates a list of n random float number between 0 and 1
    total = sum(percentages)  # Computes the accumulated percentage sum of list elements
    return {i: p / total for i, p in zip(RANGE, percentages)}  # Normalisation such that the sum amounts to 1

def generate_demand_bids(RANGE, mean=1, std=0.5):
    # Generate n normally distributed integers
    return {i: abs(random.gauss(mean, std)) for i in RANGE}  # Store in a dictionary with indices as keys


# Define ranges
GENERATORS = ['W1','W2','W3','G1','G2'] #range of generators
LOADS = [f'D{i}' for i in range(1,n_load+1)] #range of loads (D1,...)
BESSs = ['B1','B2','B3'] #range of BESSs (B1,...)
TIMES = [t for t in range(1, 25)] # 24 hour time-steps] #range of Time steps between 1,...,24
         
         
# Define input parameters of loads and generators
generator_cost = {'W1':0,'W2':0,'W3':0,'G1':100,'G2':40} # Generators costs
generator_capacity = {'W1':150,'W2':100,'W3':50,'G1':500,'G2':100} # Generators capacity (overline{P}^G_i)
generator_availability = {
    'W1': {1: 0.384460432271812, 2: 0.334138265321714, 3: 0.39211021799672, 4: 0.320718432808069, 5: 0.511097833226585, 6: 0.670195008873801, 7: 0.732582856766447, 8: 0.715879043358315, 9: 0.81648374993383, 10: 0.863173498473644, 11: 0.834677137635638, 12: 0.809602492154417, 13: 0.779704415873997, 14: 0.737250632940909, 15: 0.720228322487059, 16: 0.74521023455785, 17: 0.682319241783506, 18: 0.656484829180529, 19: 0.734256359587207, 20: 0.72407359526009, 21: 0.736487840564778, 22: 0.631564115077659, 23: 0.624393969124891, 24: 0.689311008256275}, 
    'W2': {1: 0.563373256644606, 2: 0.556427015328695, 3: 0.62006417150311, 4: 0.565904840310144, 5: 0.662962577330435, 6: 0.672047757071126, 7: 0.683555715690049, 8: 0.690681448075447, 9: 0.706195409946672, 10: 0.655940860017876, 11: 0.68401154022735, 12: 0.701908509163195, 13: 0.707423438938512, 14: 0.705133604858192, 15: 0.744796435626026, 16: 0.763055085806633, 17: 0.727549023726404, 18: 0.695863325033817, 19: 0.724752795465507, 20: 0.771747268782961, 21: 0.824356631287494, 22: 0.782885432728708, 23: 0.815080918457579, 24: 0.75197694599735}, 
    'W3': {1: 0.362613676602771, 2: 0.527927185676463, 3: 0.53357862582856, 4: 0.619152657560184, 5: 0.68612903725525, 6: 0.70974387706599, 7: 0.720573889475637, 8: 0.710547365158622, 9: 0.714516933056123, 10: 0.68126689566097, 11: 0.707796211142351, 12: 0.671199507398572, 13: 0.633474188784057, 14: 0.68557495287973, 15: 0.691890864192462, 16: 0.743445055398032, 17: 0.695043386042528, 18: 0.657822627338983, 19: 0.772778287545165, 20: 0.802106089261677, 21: 0.835111761001025, 22: 0.825193141814975, 23: 0.821347810888441, 24: 0.772766234701015},
    'G1': {t: 1 for t in TIMES},
    'G2': {t: 1 for t in TIMES}}  # Controllable generators are always available while wind turbines availability is determined by (wind scenario='V1', time step)
load_cost = 2000 # EUR/MWh
load_cost_normalized = generate_demand_bids(LOADS) # Generates bids for every loads (keeping the same bid profile for every hour)
load_bids = {d: load_cost*load_cost_normalized[d] for d in LOADS}
load_capacity = 400 # Total load capacity (overline{P}^D_i)
load_distribution = generate_demand_function(n_load, LOADS)
load_profile_IEEE = {
    1: 1775.835,
    2: 1669.815,
    3: 1590.3,
    4: 1563.795,
    5: 1563.795,
    6: 1590.3,
    7: 1961.37,
    8: 2279.43,
    9: 2517.975,
    10: 2544.48,
    11: 2544.48,
    12: 2517.975,
    13: 2517.975,
    14: 2517.975,
    15: 2464.965,
    16: 2464.965,
    17: 2623.995,
    18: 2650.5,
    19: 2650.5,
    20: 2544.48,
    21: 2411.955,
    22: 2199.915,
    23: 1934.865,
    24: 1669.815} # load profile of IEEE 24-bus system
load_capacity_IEEE = max([load_profile_IEEE[t] for t in TIMES]) # max load of IEEE 24-bus system
load_profile_normalized = {t: load_profile_IEEE[t]/load_capacity_IEEE for t in TIMES} # normalized load profile between (0,1)
load_profile = {(d,t): load_distribution[d]*load_profile_normalized[t]*load_capacity for d in LOADS for t in TIMES}


# Define input parameters of batteries
BESS_cost = {g: 3 for g in BESSs}
BESS_soc_capacity = {'B1':300,'B2':300,'B3':150} # Battery maximum SOC (\overline{SOC}_i)
BESS_soc_init = {'B1':0.5,'B2':0.5,'B3':0.5} # Battery initial SOC (SOC^{init}_i)
BESS_ch_dis_capacity = {'B1':0.5,'B2':0.25,'B3':0.15} # Battery maximum charging/discharging (\overline{P}^{ch/dis}_i)
BESS_ch_eff = {'B1':0.9,'B2':0.9,'B3':0.9} #Battery charging efficiency (\rho^{ch}_i)
BESS_dis_eff = {'B1':1.1,'B2':1.1,'B3':1.1} #Battery discharging efficiency (\rho^{dis}_i)

