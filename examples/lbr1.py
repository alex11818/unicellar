
"""
Example based on LBR-1 reservoir (DOI:10.1016/j.egypro.2017.03.1712)
"""

# %%
import os
import unicellar as uc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime

#%% Case set up
fdir = os.path.dirname(__file__) # directory of the file

res1 = uc.Reservoir(k=100, h=5, p0=113, poro=0.25, cf = 1e-4, 
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

# Fluids:
lbr1_pvt = pd.read_csv(fdir + r".\data\lbr1_pvt.csv")

oil_pvt = uc.Fluids.Oil(fvf_table=lbr1_pvt[['pressure (bar)','oil FVF (rm3/sm3)']], 
                     rs_table =lbr1_pvt[['pressure (bar)','Rs (sm3/sm3)']])
gas_pvt = uc.Fluids.Gas(fvf_table=lbr1_pvt[['pressure (bar)','gas FVF (rm3/sm3)']])
wat_pvt = uc.Fluids.Water(cw=4e-5)
pvt = uc.Fluids(oil=oil_pvt, gas=gas_pvt, water=wat_pvt)

#  pressure measurements
pm_table = pd.read_csv(fdir + r".\data\lbr1_pressure_measurements.csv")
pm = uc.PressureMeasurements(data=pm_table, 
                                date_format='dd.mm.yyyy', 
                                start_date=datetime.datetime(1957,1,1))

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
                      title='LBR-1 cases')
fig.show(renderer='browser')
