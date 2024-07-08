# spell check
# cSpell:enable
# cSpell:includeRegExp #.*
# cSpell:includeRegExp /(["]{3}|[']{3})[^\1]*?\1/g

# plotting functions
# @author: Alexey Khrulenko (alkh@norceresearch.no)

import plotly.graph_objects as go
# import pandas as pd
import numpy as np
import copy

def multiplot(df, vectors, axes={}, x='index', fig = None,
            write_html=None, renderer=None, **layout_params):
    ''' configurable multiline plot in Plotly
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with data
        
    vectors : dict  
        to specify line properties as plotly dictionaries  
        For marker properties refer to https://plotly.com/python/marker-style/  
        For line properties refer to  
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Line
        For colors refer to https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
        Example:         
        vectors = {  
            'WWIR_W1':
            {'yaxis': 'y', 'name': 'rate',  
             'mode': 'lines+markers',  
             'marker': {'color': 'royalblue', 'size': 4,  
                        'symbol': 'circle-open'},  
             'line': {'width': 2}},  
            'WBHP_W1':  
            {'yaxis': 'y3', 'name': 'BHP',  
             'mode': 'lines',  
             'line': {'color': 'red'},  
             'filter': "df['WWIR_W1']>100"}  
        }  
        Take a note that 'filter' record may be used to specify filtering   
        as shown above.  
        
    axes : dict  
        to specify axes properties  
        For colors refer to https://developer.mozilla.org/en-US/docs/Web/CSS/named-color  
        Example:       
        axes = {  
            'y': {'title': 'rate', 'color': 'dodgerblue'},     
            'y2': {'title': 'pressure', 'color': 'red'},    
            'y3': {'title': 'abs(dp/dt)', 'color': 'orange'},       
        }        
        
    x : str, optional  
        vector to be used as x-argument   
        if 'index' => df.index is used    
    
    fig : plotly figure  
        plotly figure to be used to add traces (see example below)   
        
    renderer : str, optional    
         to immidiately render the plot, options:    
         * 'browser' - in browser    
         * 'jupyterlab' inline in jupyterlab    
         
    write_html : str, None, optional  
        path to save html with the chart  
    
    layout_params : dict, optional  
        plotly layout parameters (refer plotly.com) to overwrite default ones  
        Example:    
        dict(font_size=16, template='plotly_white', title='all-in-one chart',  
             legend=dict(yanchor="middle", y=0.5, xanchor="center", x=1.07,  
             groupclick="toggleitem"))        
    
    Returns  
    -------  
    plotly figure  
        
    Example  
    --------
    vd1 = {  
        'WWIR_W1': {'axis': 'y', 'name': 'rate',  
                    'mode': 'lines',  
                    'marker': {'color': 'royalblue', 'size': 6, 'symbol': 'circle-open'},  
                    'line': {'width': 2},  
                    'filter': "df["WWIR_W1"]>0"    # filter can be used  
                    },  
         }  
    ad1 = {'y': {'title': 'rate', 'color': 'dodgerblue'}}  
    fig = aw_multiplot_(R_[0],x='index', vd=vd1, ad=ad1)

    vd2 = {  
        'WBHP_W1': {'axis': 'y2',  'mode': 'lines',  
                    'name': 'BHP','color': 'red',   
                    'line': {'color': 'red'},}  
         }  
    ad2 = {'y2': {'title': 'pressure', 'color': 'red'}}  
    fig = aw_multiplot_(R_[0],x='index', vd=vd2, ad=ad2, renderer='browser', fig=fig)  
    fig   
    '''
    if fig is None:
        fig = go.Figure()   
    else:
        # to avoid changes in fig
        fig = copy.deepcopy(fig)
        
    vd = copy.deepcopy(vectors)
    ad = copy.deepcopy(axes)
      
    for n,v in enumerate(vd):  
        
        xx = df.index if x=='index' else df[x]
        yy = df[v]
        
        # back compatibility
        if vd[v].get('axis') is not None:
            vd[v]['yaxis'] = vd[v].pop('axis')
        
        if vd[v].get('filter') is not None:
            fltr = vd[v].get('filter', None)
            fltr = eval(fltr)
            xx = xx[fltr]
            yy = yy[fltr]
            del vd[v]['filter'], fltr

        vd[v]['name'] = vd[v].get('name', v)
        
        fig.add_trace(
            go.Scattergl(
                x = xx, y = yy, 
                **vd[v]
            ))       
    
    layout_dict = fig['layout'].to_plotly_json()
    yaxis = layout_dict.get('yaxis',{})
    if 'y' in ad:
        yaxis = {**yaxis, **ad['y']}
        if yaxis.get('titlefont') is None: 
            yaxis['titlefont']={'color': yaxis.get('color')}
        if yaxis.get('tickfont')  is None: 
            yaxis['tickfont']= {'color': yaxis.get('color')}        
        
    yaxis2 = layout_dict.get('yaxis2',{})
    if 'y2' in ad:
        defaults=dict(anchor="free", overlaying="y", \
                      side="right", position=1.0, showgrid = False) 
        yaxis2 = {**defaults, **yaxis2, **ad['y2']}
        if yaxis2.get('titlefont') is None: 
            yaxis2['titlefont']={'color': yaxis2.get('color')}
        if yaxis2.get('tickfont')  is None: 
            yaxis2['tickfont']= {'color': yaxis2.get('color')}
       
    yaxis3 = layout_dict.get('yaxis3',{})
    if 'y3' in ad:
        defaults=dict(anchor="free", overlaying="y", \
                      side="right", position=0.0, showgrid = False)
        yaxis3 = {**defaults, **yaxis3, **ad['y3']}
        if yaxis3.get('titlefont') is None: 
            yaxis3['titlefont']={'color': yaxis3.get('color')}
        if yaxis3.get('tickfont')  is None: 
            yaxis3['tickfont']= {'color': yaxis3.get('color')}
        
    yaxis4 = layout_dict.get('yaxis4',{})
    if 'y4' in ad:    
        defaults=dict(anchor="free", overlaying="y", side="left", \
                      position=1.0, showgrid = False)
        yaxis4 = {**defaults, **yaxis4, **ad['y4']}
        if yaxis4.get('titlefont') is None: 
            yaxis4['titlefont']={'color': yaxis4.get('color')}
        if yaxis4.get('tickfont')  is None: 
            yaxis4['tickfont'] ={'color': yaxis4.get('color')}
        
    fig.update_layout(
        template='plotly_white',    
        yaxis = yaxis, yaxis2 = yaxis2, yaxis3 = yaxis3, yaxis4 = yaxis4,
        # hovermode = 'x unified',
        modebar_add=['toggleHover','drawline', 'drawopenpath', 'drawclosedpath',
                     'drawcircle', 'drawrect','eraseshape','toggleSpikelines'], 
        legend=dict(orientation="v", yanchor="middle", \
                    y=0.5, xanchor="left", x=1.07, groupclick="toggleitem"),
        # font_size = 14,    
    )
    
    # user-defined layout parameters
    fig.update_layout(layout_params)

    if isinstance(write_html,str):
        fig.write_html(write_html)

    if renderer is not None:
        fig.show(renderer=renderer)

    return fig


def plotly_chart(cases, renderer=None, \
    xaxis = 'date', rate_mode = 'rc',
    colors = None, marker = None,
    show_aquifer_pressure=False,    
    show_description=True,
    show_dynamic_pressures='legendonly',
    show_net_withdrawal=True,       
    show_wells = False,
    **layout_params):
    '''
    creates 'all-in-one' Plotly multichart of production profile and pressures

    Parameters
    ----------
    cases : unicellar.Case, list of unicellar.Case instances, 
        dict like {"case_name1": unicellar.Case, "case_name2": unicellar.Case, ...}
        unicellar.Case instance(s) to be visualized
        NB! production history and pressure measurements are visualized only 
        for the first case

    renderer : str
        use to visualize plot:
        fig.show(renderer='browser')

    xaxis : str, optional
        xaxis='date' => everything plotted vs. date
        xaxis='days' => everything plotted vs. days
        xaxis='years' => everything plotted vs. years
    
    rate_mode : str, optional
        options to visualize rates
        * 'RC' - rates in reservoir conditions
        * 'mass' - mass rates
        * 'SC' - rates in surface conditions

    colors: dict, optional
        sets colors to fluid rates and cases. Defaults:
        colors = \
            {'rates':
                {'oil_prod': 'green', 
                'wat_prod': 'blue', 
                'gas_prod': 'red', 
                'co2_prod': 'mediumvioletred',
                'oil_inj':  'darkgreen', 
                'wat_inj': 'darkblue', 
                'gas_inj': 'tomato', 
                'co2_inj': 'blueviolet',
                'aquifer': 'dodgerblue'},
            'cases': ['darkslategray', 'crimson', 'limegreen', 'turquoise',\
                 'orange', 'magenta']
            }

    marker : dict, optional
        sets some shared marker parameters (https://plotly.com/python/marker-style/)
        * marker['color'] sets marker face color for 'circle', 'square' etc. markers
            and line color (for '-open' markers)
        * marker['line']['color'] sets marker line color for 'circle', 'square' etc. markers  
        Default:
        marker = {'size': 8, 'color': 'rgba(0,0,0,0.3)', \
            'line': {'width': 1, 'color': 'rgb(60,60,60)'}},\   

    show_aquifer_pressure : bool, optional
        display aquifer pressure(s)

    show_descriptions : bool, optional
        display case descriptions as hover text

    show_dynamic : bool, 'legendonly', optional
        * True/False show or not dynamic pressures  
        * 'legendonly' hide them intially

    show_net_withdrawal : bool, optional
        show net withdrawal rate @ rate_mode=='rc' 

    show_wells : bool
        use individual markers for wells      

    **layout_params : dict, optional
        plotly layout parameters (refer plotly.com) to overwrite default ones
        Examples:
        layout_params={'height': 700}
            or
        dict(font_size=14, template='plotly_white', \
            legend=dict(yanchor="top", y=-0.05, xanchor="center", \
                        x=0.5, groupclick="toggleitem", orientation='h'))
        Default layout parameters are:
        default_layout_params = \
            {
            'template': 'plotly_white',
            'margin': {'b': 0, 'l': 0, 'r': 0,'t': 10},
            'modebar_add': ['toggleHover','drawline', 'drawopenpath', \
            'drawclosedpath','drawcircle', 'drawrect','eraseshape',
            'toggleSpikelines']
            }
                
    Returns
    -------
    fig : plotly dict
        with two axes : 1) left ('y'): rates; 2) right ('y2'): pressures

        Some shortcuts:
        - to visualize inline: fig.show() 
        - to visualize in the browser: fig.show(renderer='browser') 
        - to write HTML: fig.write_html(file = 'path.html')

    Hints for adding new traces:
    
    1. A new trace can be added like a normal plotly trace
    2. available yaxes: "y" (fluid rates), "y2" (pressures)
    3. list of available legendgroups: 
       * "fluid rates" (stackgroup=1 for production and =1 for inj. rates)
       * "simulated pressures"
       * "measured pressures"
       * "estimated pressures" 
       * "aquifer influxes"
       * "aquifer pressures"

    4. Examples:
    # to generate figure
    fig = unicellar.plotly_chart(a_case)
    # to add a trace
    fig.add_trace(go.Scatter(
        x = x, y = y, mode = 'lines',
        name = 'name', legendgroup="new", legendgrouptitle_text='new group',
        line={'width': 3, 'color': 'black']}, yaxis = 'y2'
        ))   

    # or via multiplot function (see the description):
    ad = {'y3': {'title': 'GIP (sm3)', 'color': 'red'}}
    vd = {
        'GIP (sm3)': 
        {'axis': 'y3', 'name': 'a case',
        'mode': 'lines', 
        'legendgroup': 'GIP', 'legendgrouptitle_text': 'GIP:',  
        'line': {'width': 3, 'color': 'red', 'dash': 'solid'}
        },
    }
    fig=unicellar.multiplot(a_case.results, vd=vd, ad=ad, fig=fig1, x='date')
    fig.show(renderer='browser')
    '''
    if marker is None:
        marker = {
            'size': 8, 'color': 'rgba(0,0,0,0.3)', 
            'line': {'width': 1, 'color': 'rgb(60,60,60)'}
            }

    if colors is None:
        colors = \
            {'rates':\
            {'oil_prod': 'green', 
            'wat_prod': 'blue', 
            'gas_prod': 'red', 
            'co2_prod': 'mediumvioletred', \
            'oil_inj':  'green', 
            'wat_inj': 'blue', 
            'gas_inj': 'red', 
            'co2_inj': 'blueviolet',\
            'aquifer': 'dodgerblue'},\
            #   'cases': ['darkslategray', 'crimson', 'limegreen', 'turquoise',\
            #              'orange', 'magenta']
            'cases': ['darkslategray', 'crimson', 'lime', 'turquoise',\
                        'orange', 'magenta']        
                }

    rates_clrs= colors['rates']
    case_clrs = colors['cases']

    # if 'layout_parameters' are set as layout_parameters={...}
    default_layout_params = \
        {
         'template': 'plotly_white',
         'modebar_add': ['toggleHover','drawline', 'drawopenpath', 
         'drawclosedpath','drawcircle', 'drawrect','eraseshape',
         'toggleSpikelines']
        #  'margin': {'b': 0, 'l': 0, 'r': 0,'t': 10},         
        }

    well_markers = [\
        'circle', 'square', 'diamond', 'cross', 'x', 'star', 'hexagram',\
        'circle-cross', 'square-cross','diamond-cross', 'hash',
        ]

    # to avoid calling unicellar.Case
    if isinstance(cases, dict):
        cases_dict = cases
    elif isinstance(cases, list):
        cases_dict = {}
        names0 = []
        names = []
        for n,cs in enumerate(cases):
            # renaming duplicated names if any ...
            if cs.name is None:
                name = n
            else:
                name = cs.name
            names0.append(name)
            if names0.count(name)>1:
                name = f"{name} ({names0.count(name)})"
            names.append(name)
            cases_dict[name] = cs
        del names0, names            
    else:
        # if a just one Case instance was passed
        cases_dict = {0: cases}
    
    rate_mode = rate_mode.lower()
    if rate_mode not in ['sc','rc','mass']:
        raise ValueError(f'rate mode {rate_mode.upper()} is not known! Valid options: "SC","RC" or "mass"')

    n_cases = len(cases_dict)
    selcase = list(cases_dict.keys())[0] # case selected to visualize the parameters

    col_dict = {
        'oil_prod': 'oil prod.', 
        'wat_prod': 'wat. prod.', 
        'gas_prod': 'gas prod.', 
        'co2_prod': 'CO2 prod.',
        'oil_inj': 'oil inj.',   
        'wat_inj': 'wat. inj.',    
        'gas_inj': 'gas inj.', 
        'co2_inj': 'CO2 inj.'
        }        

    dens_dict = {}
    if cases_dict[selcase].fluids.oil is not None:
        dens_dict['oil'] = cases_dict[selcase].fluids.oil.density
    if cases_dict[selcase].fluids.water is not None:
        dens_dict['wat'] = cases_dict[selcase].fluids.water.density
    if cases_dict[selcase].fluids.gas is not None:
        dens_dict['gas'] = cases_dict[selcase].fluids.gas.density
    if cases_dict[selcase].fluids.co2 is not None:
        dens_dict['co2'] = cases_dict[selcase].fluids.co2.density

    fig = go.Figure()
    addtext = '' if n_cases==1 else f" (#1)"
    
    # adding historical rates
    if xaxis not in ['date','days','years']:
        raise ValueError(f"xaxis=={xaxis}! xaxis must be either 'date',or 'days' or 'years'")

    # t = cases[selcase].results[xaxis].T.values
    t = cases_dict[selcase].flows.rates[xaxis].T.values
    t = np.repeat(t,2)[:-1]    
    df = cases_dict[selcase].flows.rates.copy()  
    cols = df.columns
    for mm,mm2,mult in zip(['prod','inj'], ['production','injection'],[1,-1]):    
        for fl,fl2 in zip(['oil','wat','gas','co2'], ['oil','water','gas','CO2']):
            
            c = f'{fl}_{mm}'
            if (c in cols) and (not (df[c]==0).all()):
                y = df[c]
                if rate_mode == 'rc':
                    c_fvf = f'{fl2} FVF (rm3/sm3)'                    
                    y *= cases_dict[selcase].results[c_fvf]
                elif rate_mode == 'sc':
                    pass
                else: # mass
                    y *= dens_dict[fl]/1000
                
                y = y.T.values
                y = mult*np.repeat(y,2)[1:]

                fig.add_trace(go.Scatter(
                        x = t, y = y, 
                        name = col_dict[c],  mode = 'lines', 
                        legendgroup="fluid rates", 
                        legendgrouptitle_text= f'fluid rates {addtext}:',
                        line={'width': 1, 'color': rates_clrs[c]},
                        stackgroup = mult,
                        yaxis='y')
                        )        
    # adding net withdrawal rate
    if show_net_withdrawal & (rate_mode == 'rc'):
        nwr = cases_dict[selcase].results['net withdrawal rate (rm3/day)']
        nwr = np.repeat(nwr.T.values,2)[1:]
        fig.add_trace(
            go.Scatter(x = t, y = nwr,
                    name='net withdrawal',
                    mode='lines', legendgroup="fluid rates",
                    line = dict(color='black',dash='solid', width=1),
                    yaxis='y'
                    ))      

    # measured and estimated pressures
    # assumed that all cases share the same set of pressure measurements
    if cases_dict[selcase].pressure_measurements is not None:
        df = cases_dict[selcase].pressure_measurements.data.copy()
        # df['symbol'] = None
        df['hovertext'] = None
        for i in df.index:
            hovertext = None
            if df.loc[i,'well'] is not None:
                hovertext = df.loc[i,'well']
            if (df.loc[i,'comment'] is not None):
                if hovertext is not None:
                    hovertext += '<br />' + df.loc[i,'comment']
                else:
                    hovertext = df.loc[i,'comment']
            df.loc[i,'hovertext'] = hovertext

        # if show_wells and (df['well']==None).any():
        if show_wells:

            well_list = list(df['well'].unique())
            well_markers *= round(len(well_list)/len(well_markers))+1
            well_smb_dict = {}
            for w, smbl in zip(df['well'].unique(), well_markers):
                well_smb_dict[w] = smbl
            
            df['symbol'] = df['well'].map(well_smb_dict)
            
            for i in df.index:
                est = df.at[i,'estimate']  
                mtype = df.at[i,'type']
                if mtype == 'dynamic':
                    if est == False:
                        df.at[i,'symbol'] += '-open'
                    elif est == True:
                        df.at[i,'symbol'] = 'diamond-tall' 
                    elif est == 'max':
                        df.at[i,'symbol'] = 'triangle-up' + '-open-dot'
                    elif est == 'min':
                        df.at[i,'symbol'] = 'triangle-down' + '-open-dot'
                    else:
                        pass  
                else:
                    if est == False:
                        pass
                    elif est == True:
                        df.at[i,'symbol'] = 'diamond-tall-dot'
                    elif est == 'max':
                        df.at[i,'symbol'] = 'triangle-up-dot'
                    elif est == 'min':
                        df.at[i,'symbol'] = 'triangle-down-dot'
                    else:
                        pass  
                    
            for est in [False, 'max', True, 'min']:
                
                lgt_text='estimated pressures:'
                ing = '' # ending
                if est == False:
                    lgt_text='measured pressures:'
                elif est == True:   ing = ' (mean)'
                elif est == 'max':  ing = ' (max)'
                elif est == 'min':  ing = ' (min)'
                else:
                    pass        

                for mtype in ['static', 'dynamic']:

                    name = mtype + ing
                    ind = (df['estimate']==est) & (df['type']==mtype)

                    mrkr = marker.copy()     
                    if mtype == 'static':
                        visible = True
                        mrkr['color'] = 'rgba(0,0,0,0.3)'
                    else:
                        mrkr['color'] = mrkr['line']['color'] 
                        visible = show_dynamic_pressures

                    mrkr['symbol'] = df.loc[ind,'symbol']  
                    
                    if ind.any():
                        fig.add_trace(
                            go.Scatter(
                                x = df.loc[ind, xaxis], 
                                y = df.loc[ind,'value'],
                                name = name, mode = 'markers',
                                legendgroup = lgt_text.replace(':',''),                        
                                hovertext = df.loc[ind,'hovertext'], 
                                visible = visible, 
                                marker=mrkr,
                                legendgrouptitle_text = lgt_text,
                                yaxis='y2')
                        ) 
        else:
            
            for est in [False, 'max', True, 'min']:

                lgt_text='estimated pressures:'
                ing = '' # ending
                if est == False:
                    lgt_text='measured pressures:'
                    visible = True
                    smbl = 'circle'
                    ing = ''
                elif est == True:
                    smbl = 'diamond-tall'
                    ing = ' (mean)'
                elif est == 'max':
                    smbl = 'triangle-up'
                    ing = ' (max)'
                elif est == 'min':
                    smbl = 'triangle-down'
                    ing = ' (min)'
                else:
                    pass              
                
                for mtype in ['static', 'dynamic']:

                    mrkr = marker.copy() 
                    mrkr['symbol'] = smbl
                
                    if mtype == 'static':
                        mrkr['color'] = 'rgba(0,0,0,0.3)'
                        visible = True
                    else:
                        mrkr['color'] = mrkr['line']['color'] 
                        mrkr['symbol'] += '-open'                
                        visible = 'legendonly'

                    if est != False: mrkr['symbol'] += '-dot'
                    
                    name = mtype + ing
                    ind = (df['estimate']==est) & (df['type']==mtype)

                    if ind.any():
                        fig.add_trace(
                            go.Scatter(
                                x = df.loc[ind, xaxis], 
                                y = df.loc[ind,'value'],
                                name = name, mode = 'markers',
                                legendgroup = lgt_text.replace(':',''),                        
                                hovertext = df.loc[ind,'hovertext'], 
                                visible = visible, marker=mrkr,
                                legendgrouptitle_text = lgt_text,
                                yaxis='y2')
                        )  

    # simulated pressures
    for n, k in enumerate(cases_dict.keys()):
        clr = case_clrs[n]
        # print(cases[k].description if (show_description and n_cases>1) else None)
        fig.add_trace(
            go.Scatter(
                x = cases_dict[k].results[xaxis],
                y = cases_dict[k].results['pressure (bar)'],
                name = f'#{n+1}. {k} ' if n_cases > 1 else 'sim. pressure', 
                mode='lines',
                legendgroup = "simulated pressures",
                hovertext = \
                    cases_dict[k].description if (show_description and n_cases>1) else None,
                legendgrouptitle_text=\
                    "simulated pressures:" if n_cases > 1 else None,
                line = dict(color=clr, dash='solid', width=2),
                yaxis='y2')
                )   
        
    # aquifer influx 
    if n_cases == 1:
        # for single case
        if cases_dict[selcase].aquifer is not None:
            ai = cases_dict[selcase].results['aquifer influx rate (sm3/day)'].copy()            
            if rate_mode == 'rc':
                ai *= cases_dict[selcase].results['water FVF (rm3/sm3)']
            elif rate_mode == 'sc':
                pass
            else: # mass
                ai *= dens_dict['wat']/1000

            ai = -np.repeat(ai.T.values,2)[1:]

            t = cases_dict[selcase].results[xaxis].T.values
            t = np.repeat(t,2)[:-1]                
            # aquifer influx
            fig.add_trace(
                go.Scatter(
                    x = t, y = ai,
                    name = f'aquifer influx{addtext}',
                    mode = 'lines', legendgroup="fluid rates",
                    line=dict(width=1, color=rates_clrs['aquifer']),
                    stackgroup = -1,
                    yaxis='y')
                    )
            
            # aquifer pressure
            if show_aquifer_pressure:
                fig.add_trace(go.Scatter(
                        x = cases_dict[selcase].results[xaxis],
                        y = cases_dict[selcase].results['aquifer pressure (bar)'],
                        name='aquifer pressure',
                        mode='lines', legendgroup="aquifer pressures",
                        line = dict(color=case_clrs[0], dash='dash', width=1),
                        yaxis='y2')
                        )               
    else:
        # for many cases
        for n, k in enumerate(cases_dict.keys()):
            if cases_dict[k].aquifer is not None:
                clr = case_clrs[n]
                
                t = cases_dict[k].results[xaxis].T.values
                t = np.repeat(t,2)[:-1]   
                
                ai = cases_dict[k].results['aquifer influx rate (sm3/day)'].copy()
                if rate_mode == 'rc':
                    ai *= cases_dict[k].results['water FVF (rm3/sm3)']
                elif rate_mode == 'sc':
                    pass
                else: # mass
                    ai *= dens_dict['wat']/1000              

                ai = -np.repeat(ai.T.values,2)[1:]       
                fig.add_trace(
                    go.Scatter(
                        x = t, y = ai,
                        name = f'#{n+1}. {k}', 
                        mode='lines',
                        legendgroup = "aquifer influxes",
                        hovertext = \
                            cases_dict[k].description if (show_description and n_cases>1) else None,
                        legendgrouptitle_text="aquifer influxes:",
                        line = dict(color=clr, dash='dot', width=2),
                        yaxis='y'
                        ))

                # aquifer pressure
                if show_aquifer_pressure:
                    fig.add_trace(go.Scatter(
                            x = cases_dict[k].results[xaxis],
                            y = cases_dict[k].results['aquifer pressure (bar)'],
                            name = f'#{n+1}. {k}',
                            mode='lines', legendgroup="aquifer pressures",
                            legendgrouptitle_text="aquifer pressures:",
                            line = dict(color=clr, dash='dash', width=1),
                            yaxis='y2'
                            ))        

    yaxis = {}
    if rate_mode == 'rc':
        yaxis['title'] = "rates @ RC (rm3/day)"
    elif rate_mode=='sc':
        yaxis['title'] = "rates @ SC (sm3/day)"       
    else:
        yaxis['title'] = "mass rates (t/day)"

    yaxis2 = dict(
        title="pressures (bar)",
        anchor="free", 
        overlaying="y",
        side="right", position=1.0,
        showgrid = False,
    )     

    fig.update_layout(
        yaxis = yaxis,
        yaxis2 = yaxis2,
        modebar_add=['toggleHover','drawline', 'drawopenpath', 'drawclosedpath',\
            'drawcircle', 'drawrect','eraseshape','toggleSpikelines'],
        legend=dict(
            orientation="v", yanchor="middle", 
            y=0.5, xanchor="left", x=1.07,        
            bgcolor = 'rgba(255,255,255,0.2)',
            traceorder='grouped',
            groupclick="toggleitem"
            )        
        )

    if xaxis != 'date': fig.update_xaxes(title_text=xaxis)
    fig.update_layout(default_layout_params)
    
    fig.update_layout(layout_params)

    if renderer is not None:
        fig.show(renderer=renderer)

    return fig
