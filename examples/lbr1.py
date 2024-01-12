
"""
Example based on LBR-1 reservoir study (DOI:10.1016/j.egypro.2017.03.1712):
A depleted petroleum reservoir in the Czech Republic which was produced in 
1950-1970. There are many uncertainties, particularly:
1. initial pressure 
2. aquifer parameters
3. somewhat patchy production history

This example demonstrates:
1. how to set up a SCRM model
2. how to set up sensitivity runs
3. how to estimate Ultimate Storage Capacity
"""

# %%
import os
import unicellar as uc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

#%% Case set up
fdir = os.path.dirname(__file__) # directory of the file
p0 = 113 # (bar) initial pressure
res1 = uc.Reservoir(k=100, h=5, p0=p0, poro=0.25, cf = 1e-4, 
                    stoiip=9.0087E+05, giip=1.4627E+08, wiip=3603480.0)
prod_inj_table = pd.read_csv(fdir + r"\data\lbr1_production.csv", delimiter=';')
prod_inj_table['wat_inj'] = 0
# prod_inj_table['co2_inj'] = 0

# making some more cases
res2 = res1.copy()
res3 = res1.copy()
res4 = res1.copy()
res2.p0 = 116.5
res3.p0 = 120
res4.wiip = 2e+8

# Flows
flws = uc.Flows(data=prod_inj_table, date_format='dd.mm.yyyy', cumulative=False)
# adding additional empty timesteps
flws.add_rows(t=datetime(2004,1,1),num_rows=24)

# Fluids:
lbr1_pvt = pd.read_csv(fdir + r".\data\lbr1_pvt.csv")

oil_pvt = uc.Fluids.Oil(fvf_table=lbr1_pvt[['pressure (bar)','oil FVF (rm3/sm3)']], 
                     rs_table =lbr1_pvt[['pressure (bar)','Rs (sm3/sm3)']])
gas_pvt = uc.Fluids.Gas(fvf_table=lbr1_pvt[['pressure (bar)','gas FVF (rm3/sm3)']])
wat_pvt = uc.Fluids.Water(cw=4e-5)

fvf_table = uc.get_fluid_properties(\
    fluid='CO2', p_min=10, p_max=300, p_inc=10, degC=40, den_sc=1.8472)
fvf_table = fvf_table[['Pressure (bar)','FVF']]

co2_pvt = uc.Fluids.CO2(fvf_table=fvf_table)
# putting everything into the PVT model
pvt = uc.Fluids(oil=oil_pvt, gas=gas_pvt, water=wat_pvt, co2=co2_pvt)

#  pressure measurements
pm_table = pd.read_csv(fdir + r".\data\lbr1_pressure_measurements.csv")
pm = uc.PressureMeasurements(data=pm_table, 
                             date_format='dd.mm.yyyy', 
                             start_date=datetime(1957,1,1))

# aquifers
aq1 = uc.Aquifer.Fetkovich(pi=20, v=20e+8)
aq2 = uc.Aquifer.Fetkovich(pi=10, v=5e+8)
aq3 = uc.Aquifer.Fetkovich(pi=5,  v=2e+8)

lbr_case1=uc.Case(name = 'LBR-1 (strong aquifer)',\
                  reservoir=res1, flows=flws.copy(), pressure_measurements=pm, 
                  fluids=pvt, aquifer=aq1)

lbr_case2 = uc.Case(name = 'LBR-1 (medium aquifer)',
                 reservoir=res2, flows=flws.copy(), pressure_measurements=pm, 
                 fluids=pvt, aquifer=aq2)

lbr_case3 = uc.Case(name = 'LBR-1 (weak aquifer)',
                 reservoir=res3, flows=flws.copy(), pressure_measurements=pm, 
                 fluids=pvt, aquifer=aq3)

lbr_case4 = uc.Case(name = 'LBR-1 (w/o aquifer)',
                 reservoir=res4, flows=flws.copy(), pressure_measurements=pm, 
                 fluids=pvt)

lbr_case1.run(print_log=False)
lbr_case2.run(print_log=False)
lbr_case3.run(print_log=False)
lbr_case4.run(print_log=False)

fig=uc.plotly_chart([lbr_case1, lbr_case2, lbr_case3, lbr_case4], \
                      title='LBR-1 cases', show_aquifer_pressure=True)
fig.show(renderer='browser')

# %% Estimating USC. Calculating and printing out UCS tables
for case in [lbr_case1, lbr_case2, lbr_case3, lbr_case4]:
    # preparing a USC template. Historical cumulatives will be used
    case.usc = uc.usc_template(reservoir=case.reservoir, flows=case.flows)
    # some edits
    case.usc.loc[0,'p_max'] = case.usc.loc[1,'p_max']
    case.usc.loc[0,'comment'] = 'no production'
    case.usc.loc[1,'comment'] = 'after production'

    case.usc_run(print_log=False)
    print(f'\nUCS table for "{case.name}":')
    print(case.usc[['p_max', 'oil_prod', 'gas_prod', 'wat_prod', \
                    'comment', 'fluid','Rs',
                    'USC (sm3)','USC (rm3)','USC (t)']])

# %% Getting more UCS points after production and plotting them
for case in [lbr_case1, lbr_case2, lbr_case3, lbr_case4]:

    # p_max/p0    
    p_max_to_p0 = np.array([1.0, 1.05, 1.1, 1.15, 1.20, 1.25])
    # fetching and adding the last reservoir pressure
    p_last = case.results['pressure (bar)'].iloc[-1]
    if p_last<p0:
        p_max_to_p0=np.insert(p_max_to_p0, 0, p_last/p0)

    # preparing a USC template. Historical cumulatives will be used
    temp = uc.usc_template(reservoir=case.reservoir, flows=case.flows)
    case.usc = pd.DataFrame(columns=temp.columns)   
    # looping through p_max/p0
    for n,x in enumerate(p_max_to_p0):
        case.usc.loc[n,:] = temp.loc[1,:]
        case.usc.loc[n,'p_max'] = p0*x
        case.usc.loc[n,'comment']=f'p_max={x:.2f}*p0'

    case.usc_run(print_log=False)
    print(f'\nUCS table for "{case.name}":')
    print(case.usc[['p_max','oil_prod','gas_prod','wat_prod','comment', \
                    'USC (rm3)', 'USC (sm3)','USC (t)']])