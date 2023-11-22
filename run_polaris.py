import sqlite3
import os, logging, shutil
from pathlib import Path
from polarislib.runs.convergence.convergence_config import ConvergenceIteration, ConvergenceConfig
from polarislib.runs.convergence.convergence_runner import run_polaris_convergence
from polarislib.runs import polaris_runner
from polarislib.utils.cmd_runner import no_printer
from polarislib.runs.convergence.convergence_callback_functions import (
    default_end_of_loop_fn,
    default_start_of_loop_fn,
    do_nothing,
    default_async_fn,
    default_pre_loop_fn,
)

import casadi as cs
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

lista_tts = []

def run_for_splits(binary_path, project_dir, splits_dic):
    set_splits(splits_dic, project_dir / "Campo-Supply.sqlite")
    
    output_dir, scenario_file_ = polaris_runner.run(project_dir, binary_path, "scenario_traffic.json", {"seed": 1234567}, 4, no_printer, do_nothing)

    return compute_tts(output_dir)

def compute_tts(out_dir):
    conn = sqlite3.connect(out_dir/"Campo-Demand.sqlite")
    tts = conn.execute("SELECT SUM(end-start) from trip").fetchone()[0]
    conn.close()
    lista_tts.append(tts)
    return tts

def set_splits(dic_with_splits, supply_path):
    conn = sqlite3.connect(supply_path)

    timing_id_map = {}

    for timing_id, signal, phases in conn.execute("SELECT timing_id, signal, phases from timing"):
        if signal not in timing_id_map:
            timing_id_map[signal] = []
        
        timing_id_map[signal].append(timing_id)
    
    for intersection in dic_with_splits:
        for p, split in enumerate(dic_with_splits[intersection]):
            for object_id in timing_id_map[intersection]:
                conn.execute(f"update timing_nested_records set value_minimum={split}, value_maximum={split} where object_id={object_id} and value_phase={p+1}")
    conn.commit()
    conn.close()

if __name__ == '__main__':
    project_dir = Path("C:/Users/maria/Desktop/IC")
    polaris_binary = "C:/Users/maria/Desktop/POLARIS-Bloomington-Distributable-20230907/polarisbin/Integrated_Model.exe"
    tts = run_for_splits(polaris_binary, project_dir, {10417: [8, 29, 8, 22]})
    lista_tts.append(tts)
    print("tts is", tts)
    
#-----------#
# ALGORITHM #
#-----------#

def trust_region(beta, x_k, Delta_k):
    opti = cs.Opti() # optimization problem - CASADI
    
    # Decision variables
    x = opti.variable(2) # two green splits
    #x[0] = x_k[0]
    #x[1] = x_k[1]
    
    # Objective function
    opti.minimize(beta[0] + beta[1] * x[0] + beta[2] * x[1]
                          + beta[3] * x[0] ** 2 + beta[4] * x[1] ** 2)
    
    # Constraints
    opti.subject_to(x[0] + x[1] == 1.0)
    opti.subject_to((x[0] - x_k[0]) ** 2 + (x[1] - x_k[1]) ** 2 <= Delta_k ** 2)
    opti.subject_to(opti.bounded(0, x[0], 1))
    opti.subject_to(opti.bounded(0, x[1], 1))
    
    # Solver
    opti.solver("ipopt")
    sol = opti.solve()
    return sol.value(x) 

#-------------------------#
# STEP 0.  Initialization #
#-------------------------#

# Define for a given iteration k: m_k(x, y; v_k, q) as the metamodel (denoted
# hereafter as m_k(x))

def m(x, beta):
    return beta[0] + beta[1] * x[0] + beta[2] * x[1] + beta[3] * x[0] ** 2 \
                   + beta[4] * x[1] ** 2

#x_k = np.zeros((2,0)) # Set later

# as the iterate;

Delta_0 = pow(10, 3)
Delta_k = np.zeros((1, 1))
Delta_k[0, 0] = Delta_0

# as the trust region radius;

beta_k = np.ones((1, 5)) # since we only have beta, we make v_k = beta_k
                         # In the paper alpha is initialized with 1 and beta
                         # with zero...

# v_k = (alpha_k, beta_k) as the vector of parameters of m_k;

# n_k = 0 # Set later
# as the total number of simulation runs carried out up until and including
# iteration k.
# u_k = 0 # Set later
# as the number of successive trial points rejected; and
epsilon_k = 0 #????
# as the  measure of stationarity (norm of the derivative of the Lagrangian
# function of the TR subproblem with regards to endogenous variables) evaluated
# at x_k.

# The constants
eta_1 = pow(10, -3)
gamma = 0.9
gamma_inc = 1.2
epsilon_c = pow(10, -6)
tau_bar = 0.1
d_bar = pow(10, -2)
u_bar = 10
Delta_max = pow(10, 10)
# are given such that 0 < eta_1 < 1, 0 < gamma < 1 < gamma_inc, epsilon_c > 0,
# 0 < tau_bar < 1, 0 < d_bar < Delta_max, u_bar \in N*. Set the total number of
# simulation runs permitted (across all points) 
n_max = 3
# this determines the  computation budget.

#Set the number of simulation replications per point r (defined in
# Equation (2)).
r = 1

# Set
k = 0

u_0 = 0
u_k = np.zeros((1, 1))
u_k[0, 0] = u_0

n_0 = 1
n_k = np.zeros((1, 1), dtype=np.uint32)
n_k[0, 0] = n_0

# Determine x_0 and Delta_0 (Delta_0 \in (0, Delta_max])).
C = 51              #É o total de tempo verde editável
x_0 = np.zeros(2)
x_0[0] = 22/C   
x_0[1] = 29/C
x_k = np.zeros((1, 2))
x_k[0] = x_0


# Compute T and f_hat at x_0, fit an initial model m_0 (i.e., compute v_0).
# We are not using T, computing f_hat
f_hat_0 = run_for_splits(polaris_binary, project_dir, {10417: [8, x_0[0]*C, 8, x_0[1]*C]})
lista_tts.append(f_hat_0)
f_hat_k = np.zeros((1, 1))
f_hat_k[0, 0] = f_hat_0
f_hat_n_k = f_hat_k # all evaluations of  f_hat to avoid re-evaluations at fun
                    # WE COULD ALSO THINK OF A MEMOIZATION FOR M

tau_k = np.zeros((1,1))

rho_k = np.zeros((1,1))

# Fitting m_0 to obtain beta_0 (equivalent to v_0, since we do not have alpha).
# We will need x to be a two dimensional vector

x_n_k = x_k # Vector with all trial points

def fun(beta_k):
    w_0 = 0.1
    f = 0
    for i in range(n_k[k, 0]):
        w_ki = 1 / (1 + np.linalg.norm(x_k[k, :] - x_n_k[i, :]))
        f = f + (w_ki * (f_hat_n_k[i, :] - m(x_n_k[i, :], beta_k))) ** 2
    f = f + (w_0 * beta_k[0]) ** 2 + (w_0 * beta_k[1]) ** 2 \
          + (w_0 * beta_k[2]) ** 2 + (w_0 * beta_k[3]) ** 2  \
          + (w_0 * beta_k[4]) ** 2
    return f
beta_k[0] = least_squares(fun, beta_k[k, :]).x

while n_k[k,:][0] < n_max:
    #--------------------------#
    # STEP 1. Criticality step #
    #--------------------------#
    
    epsilon_k = 0 # ????
    
    if epsilon_k <= epsilon_c:
        pass # switch to conservative mode (detailed in $4.3).
     
    #--------------------------#
    # STEP 2. Step calculation #
    #--------------------------#
    
    # Compute a step s_k that reduces the model m_k and such that x_k + s_k
    # (the trial point) is in the thrust region (i.e., approximately solve the
    # TR subproblem).
    x_k_s_k = trust_region(beta_k[k, :], x_k[k, :], Delta_k[k, :])
    print('xksk', x_k_s_k, k)
    
    #---------------------------------------#
    # STEP 3. Acceptance of the trial point #
    #---------------------------------------#
    
    # Compute 
    
    f_hat_x_k_s_k = run_for_splits(polaris_binary, project_dir, {10417: [8, x_k_s_k[0]*C, 8, x_k_s_k[1]*C]})

    # and 
    
    f_hat_x_k = run_for_splits(polaris_binary, project_dir, {10417: [8, x_k[k,0]*C, 8, x_k[k,1]*C]})
    rho_k = np.vstack((rho_k, (f_hat_x_k - f_hat_x_k_s_k) / \
                      (m(x_k[k, :], beta_k[k, :]) - m(x_k_s_k, beta_k[k, :]))))

    if rho_k[k, :][0] >= eta_1: #then accept the trial point:
        print('accepted', k)
        x_k = np.vstack((x_k, x_k_s_k))
        u_k = np.vstack((u_k, u_k[k, :]))
        lista_tts.append(f_hat_x_k_s_k)
    else: # reject the trial point:
        print('rejected', k)
        x_k = np.vstack((x_k, x_k[k, :]))
        u_k[k, :] = u_k[k, :] + 1
        lista_tts.append(f_hat_x_k)
    
    # Include the new observation in the set of sampled points
    n_k[k, :] = n_k[k, :] + r
    x_n_k = np.vstack((x_n_k, x_k_s_k))
    f_hat_n_k = np.vstack((f_hat_n_k, f_hat_x_k_s_k))
    
    # update the weights w and fit the new model m_k_1.
    beta_k = np.vstack((beta_k, least_squares(fun, beta_k[k, :]).x))
    
    #---------------------------#
    # STEP 4. Model improvement #
    #---------------------------#
    
    # Compute tau_k_1 = ||v_k_1 - v_k||/||v_k||

    tau_k = np.vstack((tau_k, np.linalg.norm(beta_k[k + 1, :] \
                               - beta_k[k, :]) / np.linalg.norm(beta_k[k, :])))
    
    
    if tau_k[k+1,0] < tau_bar:
        # then improve the model by simulating the performance of a new point
        # x, which is sampled from a distribution such as the one in $4.3.
        g = np.random.uniform(low = 0.1, high = 0.9)
        new_x = np.array([g, 1 - g]) 
        # Evaluate T and f_hat at x. (we evaluate only f_hat)
        f_hat_new_x = run_for_splits(polaris_binary, project_dir, {10417: [8, new_x[0]*C, 8, new_x[1]*C]})
        lista_tts.append(f_hat_new_x)
        f_hat_n_k = np.vstack((f_hat_n_k, f_hat_new_x)) 
        # Include this new observation in the set of sampled points
        
        n_k[k, :] = n_k[k, :] + r
        x_n_k = np.vstack((x_n_k, new_x))
        
        #Update m_k_1
        beta_k[k+1, :] = least_squares(fun, beta_k[k, :]).x
        
        print('IMPROVE', k)

    
    #------------------------------------#
    # STEP 5. Trust region radius update #
    #------------------------------------#
    # Attention, numpy is not able do deal with large numbers, if Delta_k is
    # increasing close to Delta_max, anothr approach should be use. That's why
    # I'm not using the np.min but min
    if rho_k[k, :][0] > eta_1:
        Delta_k = np.vstack((Delta_k, min(gamma_inc * Delta_k[k, :][0], Delta_max)))
    elif rho_k[k, :][0] <= eta_1 and u_k[k, :][0] >= u_bar:
        Delta_k = np.vstack((Delta_k, np.min(gamma * Delta_k[k, :][0], d_bar)))
        u_k[k, :] = [0]
    else:
        Delta_k = np.vstack((Delta_k, Delta_k[k, :]))
    
    if Delta_k[k+1, :] <= d_bar:
        pass # switch to conservative mode
    
    # Set
    n_k = np.vstack((n_k, n_k[k, :]))
    u_k = np.vstack((u_k, u_k[k, :]))
    k = k + 1    
    
    # Laço while com essa condição!
    #if n_k[k,:][0] < n_max:
    #    pass # go to Step 1
    #else:
    #    pass # stop
    
#print('Done')
#print('Delta_k', Delta_k)
#print('x_n_k', x_n_k)
#print('x_k', x_k)
#print('Initial:', f_hat(x_k[0, :]))
#print('Webster:', f_hat([0.5714, 0.4286]))
#print('Webster:', f_hat([0.875, 0.125])) # algumas vezes se perde para esse cenário
#print('Optimal:', f_hat(x_k[-1, :]))

plt.plot(range(len(lista_tts)), lista_tts)
plt.figure()

plt.plot(range(len(x_n_k[:,0])), x_n_k[:,0])
plt.plot(range(len(x_n_k[:,1])), x_n_k[:,1])

plt.show()