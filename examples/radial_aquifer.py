'''
Simple case to test aquifer models and showcase plotters
'''
#%% 
import unicellar as uc
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#%% reading e100 cases
fldr = os.path.dirname(os.path.realpath(__file__))
p=[fr"{fldr}\test_e100_cases\radial_aquifer\f_1gb.data",
   fr"{fldr}\test_e100_cases\radial_aquifer\f_100gb.data"]

rr = {}
x = 'FPR'
for n,i in enumerate(p):
    cn = os.path.basename(i)[:-5]
    rr[cn] = uc.run_read_e100(i, version='2013.1', reload=True)

# er = rr['f_100gb'] # 100-cell model
e100_mdl = 'f_1gb'
# e100_mdl = 'f_100gb'

er = rr[e100_mdl] # single-cell model

#%% Setting up and running a model
# Flows (produced injected fluids)
prodInj_table = pd.DataFrame()
prodInj_table['days'] = er[0].index.values
prodInj_table['wat_prod'] = er[0]['FWPT'].values
prodInj_table['wat_inj'] = er[0]['FWIT'].values
flows = uc.Flows(data = prodInj_table, cumulative = True)
print(flows)

# Pressure measurements
# pm = pd.DataFrame()
# pm['days'] = er[0].index.values
# pm['value'] = er[0]['FPRP'].values
# pm = uc.PressureMeasurements(data = pm)

# Fluids (PVT properties)
wat_pvt = uc.Fluids.Water(cw = 4e-5)
fluids = uc.Fluids(water = wat_pvt)

# Reservoir properties
p0 = 100       # pressure, bar
poro = 0.2     # porosity
h = 10         # thickness, m
r = 1000       # outer radius, m
cw = 5e-5        # water compressibility, 1/bar
bw0 = (1+cw*(p0 - 1)) # water FVF at RC
w = 3.14*r**2*h*poro*bw0 # initial water volume
res = uc.Reservoir(cf = 1e-05, wiip = w, p0 = 100)

# Aquifer
aq = uc.Aquifer.Fetkovich(v=1e+9, pi=10)

# putting everything together into a model
mdl=uc.Case(name='SCRM, Fetkovich aq.', \
            reservoir=res, aquifer=aq, flows=flows, fluids=fluids, \
            # pressure_measurements=pm
            )

# model with alternative aquifer formulation
mdl2 = mdl.copy()
mdl2.aquifer.mode = 'modified'
mdl2.name = 'SCRM, modified Fetkovich'

mdl.run(print_log=False)
mdl2.run(print_log=False)

print(mdl.results)
print(mdl.results.columns)

# %% plotting
fig=uc.plotly_chart(
    [mdl, mdl2],  xaxis='days', show_net_withdrawal=False,\
    title='Comparison of different aquifer formulations and E100 model')

# This is how one can add another line
ad={'y3': {'title': 'cumulative aquifer influx (sm3)'}}
# adding cumulative aquifer influxes of SCRMs

vd={'cumulative aquifer influx (sm3)':\
    {'mode': 'lines','axis': 'y3',\
     'name': '#1. ' + mdl.name,
     'legendgrouptitle_text': 'cumulative aq. infl.:',
     'legendgroup': 'cumulative aq. infl.',
     'line': {'width': 3, 'color': 'dimgrey', 'dash': 'longdash'}
     }}

fig=uc.multiplot(mdl.results, vd, ad, x='days', fig=fig)

vd['cumulative aquifer influx (sm3)']['line']['color']='red'
vd['cumulative aquifer influx (sm3)']['name'] ='#2. ' + mdl2.name

fig=uc.multiplot(mdl2.results, vd, ad, x='days', fig=fig)

vd = {
    'FPR': {'axis': 'y2', 'name': f'e100 ({e100_mdl})',
    # 'mode': 'lines',
    'mode': 'markers',    
    'legendgroup': 'simulated pressures',
    'marker': {'color': 'blue', 'size': 6, 'symbol': 'circle'},
    'line': {'width': 3, 'color': 'cornflowerblue', 'dash': 'dash'}
    },        
    'FAQR': {'axis': 'y', 'name': f'e100 ({e100_mdl})',
    # 'mode': 'lines',
    'mode': 'markers',
    'legendgroup': 'aquifer influxes',
    'marker': {'color': 'blue', 'size': 6, 'symbol': 'square-open'},
    'line': {'width': 3, 'color': 'cornflowerblue', 'dash': 'dot'}
    },    
    'FAQT': {'axis': 'y3', 'name': f'e100 ({e100_mdl})',
    # 'mode': 'lines',
    'mode': 'markers',
    'legendgroup': 'cumulative aq. infl.',
    'legendgrouptitle_text': 'cumulative aq. infl.',
    'marker': {'color': 'blue', 'size': 6, 'symbol': 'diamond-open'},
    'line': {'width': 3, 'color': 'blue'}
    },
}

rr['f_1gb'][0]['FAQR'] *= -1 # negating for plotting

fig=uc.multiplot(rr['f_1gb'][0], vd, ad, fig=fig,\
                #  legend={'yanchor':"bottom", 'y': 0.05,\
                #          'xanchor': "right", 'x': 1.0,\
                #          'orientation':'h'}
                         )

# fig=uc.multiplot(rr['f_1gb'][0], vd, ad, fig=fig)


fig.show(renderer='browser')
