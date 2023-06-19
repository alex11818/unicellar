# 
"""
gas storage test/example case
This case demonstrates how ..:
1. to set up a gas properties by means of the NIST database
2. to read E100 results
3. to set up pressure measurements
4. to 
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
import unicellar as uc

#%% Flows
fldr = os.path.dirname(os.path.realpath(__file__))
p = rf"{fldr}\test_e100_cases\gas_storage\gas_storage.data"
er=uc.run_read_e100(p, reload=True)

prodInj_table = pd.DataFrame()
prodInj_table['date'] = er[0].index.values
prodInj_table['gas_prod'] = er[0]['FGPT'].values
prodInj_table['gas_inj']  = er[0]['FGIT'].values
flows = uc.Flows(data=prodInj_table, cumulative=True, date_format='datetime')
#%% Pressure measurements
t0 = flows.totals.loc[0,'date']
print(f'start date: {t0}' )

pm = pd.DataFrame()
pm['date'] = er[0].index.values
pm['value'] = er[0]['FPRP'].values
pm = uc.PressureMeasurements(data=pm, date_format='datetime', start_date=t0)

#%% Fluids
# gas PVT can be set by: 
# 1) getting CH4 properties from NIST database
# fvf_table = uc.get_fluid_properties(fluid='CH4', p_min=10, p_max=400, p_inc=10, degC=30, den_sc=0.6798)
# fvf_table = fvf_table[['Pressure (bar)','FVF']]
# 2) importing PVDG from a E100/OPM deck
fvf_table, pvdg = uc.read_pvdg(p)
# 3) using default settings

gas_pvt = uc.Fluids.Gas(fvf_table=fvf_table, density = 0.6798)
wat_pvt = uc.Fluids.Water(cw=4e-5)
fluids = uc.Fluids(water=wat_pvt, gas=gas_pvt)

#%% Reservoir
w = er[0]['FWIP'][0] # sm3
g = er[0]['FGIP'][0] # sm3
res = uc.Reservoir(cf=1e-5, wiip=w, giip=g, p0=100)
#%% Aquifer
aq = uc.Aquifer.Fetkovich(v=1e+11, pi=1000, mode='modified')
# aq = None

#%% MoDeL
mdl=uc.Case(reservoir=res, aquifer=aq, flows=flows, fluids=fluids,\
            pressure_measurements=pm)

mdl.run(print_log=True)

# plotting
fig = uc.plotly_chart([mdl], renderer='browser', rate_mode='rc')