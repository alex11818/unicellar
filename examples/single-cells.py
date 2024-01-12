'''
Two oil-water-gas depletion-repressurization tests compared with and based 
on e100 models with oil PVT defined by PVCO/PVTO keywords.  
It demonstrates:  
1. how to set up of a proxy model for e100/OPM
2. how to import of oil/gas properties from e100 models
3. validation of PVT model
4. effects of imperfect approximation of oil compressibility for undersaturated  
conditions (see difference between PVTO- and PVCO-based cases)
approximation of compressibility from PVTO  
5. how to add a new line to the plotly chart
'''

# %% 
import os

import sys
sys.path.append(r'C:\Users\alkh\OneDrive - NORCE\Python_codes\unicellar')

import unicellar as uc
import datetime
import pandas as pd
import numpy as np

# rerun=True
rerun=False

fldr = os.path.dirname(os.path.realpath(__file__))
#%%
p1 = f"{fldr}\\test_e100_cases\\single-cell\\SINGLE-CELL_PVCO.DATA"
SC1 = uc.run_read_e100(p1, version='2013.1', reload=not rerun)

p2 = f"{fldr}\\test_e100_cases\\single-cell\\SINGLE-CELL_PVTO.DATA"
SC2 = uc.run_read_e100(p2, version='2013.1', reload=not rerun)
# %% PVT
bg_table, pvdg = uc.read_pvdg(p1)
pvt_wat=uc.Fluids.Water(cw=4.0e-05, fvf_ref=1.0, p_ref=1.0, density=1000)
pvt_gas=uc.Fluids.Gas(fvf_table=bg_table, density=0.91334)

bot1, rst1, cot1, pvco = uc.read_pvco(p1)
bot2, rst2, cot2, pvto = uc.read_pvto(p2, co_est_ind=':5')

pvt_oil1=uc.Fluids.Oil(fvf_table=bot1, rs_table=rst1, co_table=cot1, density=828.37)
pvt_oil2=uc.Fluids.Oil(fvf_table=bot2, rs_table=rst2, co_table=cot2, density=828.37)

PVT1 = uc.Fluids(oil=pvt_oil1, water=pvt_wat, gas=pvt_gas)
PVT2 = uc.Fluids(oil=pvt_oil2, water=pvt_wat, gas=pvt_gas)
# %% Flows
# 1
start_date = datetime.datetime(2025,1,1)
prodInj_table1 = pd.DataFrame()
prodInj_table1['days'] = SC1[0].index
prodInj_table1['oil_prod'] = SC1[0]['FOPT'].values
prodInj_table1['oil_inj'] = 0
prodInj_table1['gas_prod'] = SC1[0]['FGPT'].values
prodInj_table1['gas_inj']  = SC1[0]['FGIT'].values
prodInj_table1['wat_prod'] = SC1[0]['FWPT'].values
prodInj_table1['wat_inj'] = SC1[0]['FWIT'].values
flows1 = uc.Flows(data=prodInj_table1, cumulative=True, start_date=start_date)
# 2
start_date = datetime.datetime(2025,1,1)
prodInj_table2 = pd.DataFrame()
prodInj_table2['days'] = SC2[0].index
prodInj_table2['oil_prod'] = SC2[0]['FOPT'].values
prodInj_table2['oil_inj'] = 0
prodInj_table2['gas_prod'] = SC2[0]['FGPT'].values
prodInj_table2['gas_inj']  = SC2[0]['FGIT'].values
prodInj_table2['wat_prod'] = SC2[0]['FWPT'].values
prodInj_table2['wat_inj'] = SC2[0]['FWIT'].values
flows2 = uc.Flows(data=prodInj_table2, cumulative=True, start_date=start_date)

# %% reservoir
rs0 = SC1[0]['FRS'][0] 
oip = SC1[0]['FOIP'][0] 
gip = SC1[0]['FGIP'][0]
wip = SC1[0]['FWIP'][0]
p0 = SC1[0]['FPR'][0]
res1=uc.Reservoir(wiip=wip, giip=gip, stoiip=oip, cf=1e-5, p0=p0, rs0=rs0)

sc_case1=uc.Case(
    reservoir=res1, fluids=PVT1, flows=flows1, name='PVCO', \
    description='SRM model based on SINGLE-CELL_PVCO.DATA'
    )
sc_case1.run()

rs0 = SC2[0]['FRS'][0] 
oip = SC2[0]['FOIP'][0] 
gip = SC2[0]['FGIP'][0]
wip = SC2[0]['FWIP'][0]
p0 = SC2[0]['FPR'][0]
res2=uc.Reservoir(wiip=wip, giip=gip, stoiip=oip, cf=1e-5, p0=p0, rs0=rs0)

sc_case2=uc.Case(
    reservoir=res2, fluids=PVT2, flows=flows2, name='PVTO', \
    description='SRM model based on SINGLE-CELL_PVTO.DATA'
    )
sc_case2.run()


#%% Plotting two cases
fig=uc.plotly_chart([sc_case1, sc_case2], xaxis='days', rate_mode='mass')

# how to add lines on the chart
vd1 = {
    'FPR': {'yaxis': 'y2', 'name': 'FPR (PVCO)', 'mode': 'markers',
            'legendgroup': 'sim. pressures (e100)',
            'legendgrouptitle_text': 'sim. pressures (e100):',
            'marker': {'color': 'grey', 'size': 5, 'symbol': 'circle-open'}, 
            'line': {'width': 2, 'dash': 'dash'}},      
     }  

fig = uc.multiplot(SC1[0], vd1, {}, x='index', fig=fig)

vd2 = vd1.copy()
vd2['FPR']['name']='FPR (PVTO)'
vd2['FPR']['marker'] = {'color': 'red', 'size': 5, 'symbol': 'diamond'}

fig = uc.multiplot(SC2[0], vd2, {}, x='index', renderer='browser', fig=fig)