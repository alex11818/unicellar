'''a small oil field (with a gas cap and a weak aquifer) goes through:
    1. depletion
    2. water injection
    3. gas injection
    4. blowdown
'''

import unicellar as uc
import datetime
import pandas as pd
import os

fldr = os.path.dirname(os.path.realpath(__file__))

# %% Reading eclipse results
# rerun=True
rerun=False

p = fr"{fldr}\test_e100_cases\oil+gas_cap+aquifer\OGA.DATA"
R = uc.run_read_e100(p, version='2013.1', reload=not rerun)

# %% flows

start_date = datetime.datetime(2025,1,1)
prodInj_table = pd.DataFrame()
prodInj_table['days'] = R[0].index
prodInj_table['oil_prod'] = R[0]['FOPT'].values
prodInj_table['oil_inj'] = 0
prodInj_table['gas_prod'] = R[0]['FGPT'].values
prodInj_table['gas_inj']  = R[0]['FGIT'].values
prodInj_table['wat_prod'] = R[0]['FWPT'].values
prodInj_table['wat_inj'] = R[0]['FWIT'].values

flows = uc.Flows(data=prodInj_table, cumulative=True, start_date=start_date)
# %% PVT
bg_table, pvdg = uc.read_pvdg(p)
bot, rst, cot, pvto = uc.read_pvto(p)

pvt_oil=uc.Fluids.Oil(fvf_table=bot, rs_table=rst, co_table=cot, density=828.37)
pvt_wat=uc.Fluids.Water(cw=4.0e-05, fvf_ref=1.0, p_ref=150.0, density=1000)
pvt_gas=uc.Fluids.Gas(fvf_table=bg_table, density=0.91334)

fvf_co2 = uc.get_fluid_properties(
    fluid='CO2', p_min=10, p_max=400, p_inc=5, degC=30, den_sc=1.8718)
fvf_co2 = fvf_co2[['Pressure (bar)','FVF']]

pvt_co2 = uc.Fluids.CO2(fvf_table=fvf_co2, density = 1.8718)

PVT = uc.Fluids(oil = pvt_oil, water=pvt_wat, gas=pvt_gas, co2=pvt_co2)

# %% case definition and run
for i in ['FOIP', 'FGIP','FGIPL', 'FWIP', 'FRS', 'FPR']:
    print(f'{i}: {R[0][i][0]} ({R[1][i]})')

rs0 = R[0]['FRS'][0] 
oip = R[0]['FOIP'][0] 
gip = R[0]['FGIP'][0]
wip = R[0]['FWIP'][0]
p0 = R[0]['FPR'][0]

res = uc.Reservoir(stoiip=oip, giip=gip, wiip=wip, cf=1e-5, p0=p0, rs0=rs0)
aq = uc.Aquifer.Fetkovich(v=2.79e+7, pi=5) # ca. 10 x reservoir PV

mdl=uc.Case(reservoir=res, fluids=PVT, flows=flows, name='GOW', 
            aquifer=aq, description='SCRM model based on GOW.DATA'
    )

mdl.run(print_log=False)

#%%  plot
fig = uc.plotly_chart([mdl], xaxis='days')

# Example of adding another line
vd = {
    'FPR': {'yaxis': 'y2', 'name': 'FPR', 
            # 'mode': 'lines+markers',
            # 'mode': 'markers',
            'mode': 'lines',
            'legendgroup': 'full-field model',
            'legendgrouptitle_text': 'full-field model',
            'marker': {'color': 'orange', 'size': 5, 'symbol': 'circle-open'}, 
            'line': {'width': 2, 'dash': 'dash'}},      
     }   

fig = uc.multiplot(R[0], vd, {}, x='index', fig=fig)
fig.show(renderer='browser')