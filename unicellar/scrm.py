# spell check
# cSpell:enable
# cSpell:includeRegExp #.*
# cSpell:includeRegExp /(["]{3}|[']{3})[^\1]*?\1/g

# -*- coding: utf-8 -*-
"""
Single-Cell Reservoir Model (SCRM)
@author: Alexey Khrulenko (alkh@norceresearch.no)
"""

import pandas as pd 
import datetime
import pickle
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from warnings import warn

import re
import copy

# %% description of variables, units and formats
_var_dict = {
    'compressibility': {'description': '', 'unit': '1/bar', 'format': '.3e'},
    'cf': {'description': 'formation (pore volume) compressibility', 'unit': '1/bar', 'format': '.3e'},
    'co': {'description': 'oil compressibility', 'unit': '1/bar', 'format': '.3e'},
    'cw': {'description': 'water compressibility', 'unit': '1/bar', 'format': '.3e'},
    'density': {'description': 'density in SC', 'unit': 'kg/sm3', 'format': ''},
    'stoiip': {'description': 'stock-tank oil initially in-place', 'unit': 'sm3', 'format': '.3e'},
    'wiip': {'description': 'water initially in-place', 'unit': 'sm3', 'format': '.3e'},
    'giip': {'description': 'gas initially in-place', 'unit': 'sm3', 'format': '.3e'},
    'co2iip': {'description': 'CO2 initially in-place', 'unit': 'sm3', 'format': '.3e'},
    'rs0': {'description': 'initial solution gas-oil ratio', 'unit': 'sm3/sm3', 'format': ''},
    'mu': {'description': 'viscosity', 'unit': 'cP', 'format': '.3e'},
    'h':  {'description': 'thickness', 'unit': 'm', 'format': '.2f'},
    'v': {'description': 'aquifer volume', 'unit': 'sm3', 'format': '.3e'},
    'pi': {'description': 'aquifer productivity index', 'unit': 'sm3/day/bar', 'format': ''},
    'k': {'description': 'permeability', 'unit': 'mD', 'format': ''},
    'poro': {'description': 'porosity in RC', 'unit': 'fraction', 'format': ''},
    'p0': {'description': 'initial reservoir pressure', 'unit': 'bar', 'format': ''},
    'p_ref': {'description': 'reference pressure', 'unit': 'bar', 'format': ''},
    'fvf_ref': {'description': 'formation vol. factor at p_ref', 'unit': '(rm3/sm3)', 'format': ''},
    'pressure': {'description': '', 'unit': 'bar', 'format': ''},
    'viscosity': {'description': 'viscosity in RC', 'unit': 'cP', 'format': ''},
    'p_inj':  {'description': 'CO2 injection pressure', 'unit': 'bar'},
    # pd.DataFrame(s)
    'totals': {'description': 'cumulative produced/injected volumes', 'unit': 'sm3'},
    'rates': {'description': 'produced/injected rates', 'unit': 'sm3/day'},
    'usc':   {'description': 'pd.DataFrame with scenarios to estimate Ultimate Storage Capacity (USC)'},
    'max_ts': {'description': 'maximal time step', 'unit': 'days', 'format': ''}
}

# %% Data for examples and defaults
# PVT from LBR-1 (DOI:10.1016/j.egypro.2017.03.1712)
pvt_lbr1=[[11.14575, 3.001414224, 1.02148336, 0.098411986],
          [21.27825, 6.382525401, 1.028739855, 0.050839815],
          [31.41075, 9.763636578, 1.03599635, 0.033974513],
          [41.54325, 13.14474776, 1.043252845, 0.02534882],
          [51.67575, 16.52585893, 1.050509341, 0.02011748],
          [61.80825, 19.90697011, 1.057765836, 0.016612418],
          [71.94075, 23.28808129, 1.065022331, 0.01410582],
          [82.07325, 26.66919246, 1.072278826, 0.012228918],
          [92.20575, 30.05030364, 1.079535321, 0.010775217],
          [102.33825, 33.43141482, 1.086791816, 0.009620054],
          [112.47075, 36.812526, 1.094048311, 0.008683484],
          [122.60325, 40.19363717, 1.101304806, 0.007911808],
          [132.73575, 43.57474835, 1.108561302, 0.0072678],
          [142.86825, 46.95585953, 1.115817797, 0.006724332],
          [153.00075, 50.3369707, 1.123074292, 0.006262251],
          [163.13325, 53.71808188, 1.130330787, 0.005865154],
          [173.26575, 57.09919306, 1.137587282, 0.005522583]]

pvt_lbr1=pd.DataFrame(data=pvt_lbr1,
                      columns=['pressure (bar)', 'Rs (sm3/sm3)',
                               'oil FVF (rm3/sm3)', 'gas FVF (rm3/sm3)'])

fvf_oil_lbr1 = pvt_lbr1[['pressure (bar)', 'oil FVF (rm3/sm3)']]
rs_of_pb_lbr1 = pvt_lbr1[['pressure (bar)', 'Rs (sm3/sm3)']]
fvf_gas_lbr1 = pvt_lbr1[['pressure (bar)', 'gas FVF (rm3/sm3)']]
del pvt_lbr1

# %% Classes ------------------------------------------------------------------
class __Au__:
    '''
    Auxiliary class to pass some basic functions/formatting to other classes
    '''
    def __init__(self):
        pass

    def __repr__(self):
        s = """"""
        # (un)comment (not) to print docstring of a class instance
        # if self.__doc__ is not None:
        #     s += self.__doc__ + '\n'

        for k, c in self.__dict__.items():
            #s += '{} \n'.format(type(vd))
            #print(k, type(c))
            vd = _var_dict.get(k, None)
            if isinstance(c, (int, float, complex, bool, str,
                              datetime.datetime,
                              pd._libs.tslibs.timestamps.Timestamp,
                              )):
                #vd = _var_dict.get(k,None)
                if vd is None:
                    s += f"{k} = {c} \n"
                else:
                    s += f"{k} = {c:{vd['format']}} ({vd['unit']}) - {vd['description']} \n"
            elif isinstance(c, (pd.DataFrame)):
                if vd is not None:
                    s += '{0} - {1}'.format(k, vd['description'])
                    un = vd.get('unit')
                    if un is not None:
                        s += f'({un})'
                    s += f':\n{c}\n'
                else:
                    s += f'{k}:\n'
                    s += f'{c}\n'
            elif c is None:
                s += '{} = {} \n'.format(k, c)
            else:
                #s += f'{k} ({type(c)}):\n'
                if vd is None:
                    s += f'{k}:\n'
                else:
                    # s += '{0} - {1} ({2}): \n'.format(k,
                    #                                   vd['description'],
                    #                                   vd['unit'])
                    s += f"{k} - {vd['description']} ({vd['unit']}): \n"
                    # if vd['unit'] is not None:
                    #     s += f'({vd['unit']})\n'
                s += f'{c}\n'
        s += f'------------ end of {self.__class__.__name__} instance ---'
        return s

    def copy(self):
        '''to make a deep copy'''
        return copy.deepcopy(self)


    def to_pickle(self, file_path):
        '''to save to a *.pkl (pickle) file'''
        if file_path[-4:] != '.pkl':
            file_path = file_path + '.pkl'
        with open(file_path, "wb") as fff:
            pickle.dump(self, fff)

class Reservoir(__Au__):
    """Class of reservoir properties

    Attributes
    ----------
    stoiip : int, float, optional 
        Standard Oil Initially In-place (sm3) 
    giip/fgiip : int, float, optional 
        (Free) Gas Initially in-place (sm3).
        Different ways may be used to initialize gas-oil cases:
        * if giip==0 and fgiip==0 => giip = rs0*stoiip
        * if giip>0 and fgiip==0 => fgiip = giip - rs0*stoiip
        * if giip==0 and fgiip>0 => giip = fgiip + rs0*stoiip
        * if giip>0 and fgiip>0 => ValueError
    wiip : int, float, optional 
        Water Initially In-place (sm3) 
    co2iip : int, float, optional 
        CO2 Initially in-place (sm3) 
    p0 : int, float, optional
        initial reservoir pressure (bar) 
    rs0 : int, float, optional 
        initial solution gas-oil ratio (sm3/sm3)

    stick_to_rs0 : bool, optional
        this option prevents free gas dissolution in the beginning if p_bub < p_res.
        The idea is to mimic behavior of commercial reservoir simulators where rs0 under gas cap
        can be set lower than the equilibrium value rs(p_res).
        For instance, in Eclipse it can be done by PBVD and RSVD keywords.
        If set to True, this option invokes the following behavior:
        1. as long as p_bub < p_res, gas dissolution is not allowed
        2. if rs becomes less rs0, then gas dissolution is allowed for all 
        subsequent timesteps
     
    compaction_model : str, function, optional 
        options to calculate pore volume multiplier (PV) as function of 
        reservoir pressure: pv=pv0*pv_mult(p0) 
        The following options are available:
            'cf' (default) - simple linear model pv_mult(p)=1-cf*(p0-p)
            <any function> that relates (p-p0) and pv_mult, example: 
                lambda p: pv0*(1+cf*(p-p0))        
            NB! This option causes an error while saving/loading pickle 
                files, converting to dict etc.
                Consider using 'pv_table' instead!!!
            'pv_table' - pv_mult is via pv_table (below): 
        NB! Regardless of the option selected the aquifer models use cf 
    pv_table : pd.DataFrame, ndarray, optional
        table of PV multipliers :
        0th column |  1st column
            p-p0   |    pv mult
             ...   |   ...
    cf : float
        formation (i.e. pore volume) compressibility (1/bar)
    k : float, optional - currently not in use, might be deprecated
        reservoir permeability (mD)        
    h :  int, float, optional - currently not in use, might be deprecated 
        reservoir thickness (m)
    poro: float, optional - currently not in use, might be deprecated
        reservoir porosity 
    print_warnings : bool
        if False => warnings are suppressed (useful for batch runs)
    """

    def __init__(self, stoiip=0, giip=0, fgiip=0, wiip=0, co2iip=0,
                 p0=None, rs0=None, stick_to_rs0=False,
                 cf=5e-5, compaction_model='cf', pv_table=None,
                 k=None, h=None, poro=0.25,
                 print_warnings=True):
        '''Initializes a class instance with the specified parameters'''
        if giip>0 and fgiip>0:
            ValueError('GIIP>0 and FGIIP>0. Only one must be defined!')

        self.stoiip = stoiip
        self.giip = giip
        self.fgiip = fgiip
        self.wiip = wiip
        self.co2iip = co2iip
        self.p0 = p0
        self.rs0 = rs0
        self.stick_to_rs0 = stick_to_rs0
        self.poro = poro
        self.cf = cf
        self.k = k
        self.h = h
        self.pv_table = pv_table
        # tests of compaction_model
        if isinstance(compaction_model, str):
            compaction_model = compaction_model.lower()
            if compaction_model == 'cf':
                pass
            elif compaction_model == 'pv table' or compaction_model == 'pv_table':
                try:
                    pv_table = copy.deepcopy(self.reservoir.pv_table)
                    if isinstance(self.reservoir.pv_table, pd.DataFrame):
                        # pv_table = pv_table.copy()
                        pv_table = pv_table.to_numpy()
                    dp = pv_table[:, 0]
                    pv_mult = pv_table[:, 1]
                    pv_mult_func = interp1d(dp, pv_mult, kind='linear')
                    if pv_mult_func(0) != 1.0 and print_warnings:
                        warn("\n   compaction_model(0)!=1.0 !!!")
                except:
                    TypeError('PV_TABLE must be ndarray or DataFrame!')
            else:
                raise ValueError(
                    f'compaction model "{compaction_model}" is unknown!')

        elif callable(compaction_model):
            if print_warnings:
                warn('\n'+
                     "   A user-defined function is passed as compaction model.\n" + \
                     "   This option may cause an error while saving/loading pickle files.\n" + \
                     "   Consider using 'pv_table' option instead!")                
                if compaction_model(0) != 1.0:
                    warn("\n   compaction_model(0)!=1.0 !!!")
        else:
            raise ValueError(f'compaction model is in a wrong format!')

        self.compaction_model = compaction_model  


class Flows(__Au__):
    '''
    Describes fluid flows in and out of the reservoir

    Attributes
    ----------
    cumulative : bool, optional  
        are volumes in DATA cumulative?

    data : pd.DataFrame
        production/injection data (see __init__ description)

    max_ts : int, float, optional 
        maximal time step (days) is used to resample DATA into TOTALS and 
        RATES attributes (original dates are retained)

    rates : pd.DataFrame
        produced/injected rates(sm3/day) calculated from DATA. 
        If MAX_TS is provided => resampling is done so that maximal time step => MAX_TS

    start_date: datetime.date object, optional
        used along with 'day' or 'year' formats mostly for plotting.
        Defaults:
        - today if date_format=='days' or 'years'
        - else: the first date in DATA

    totals : pd.DataFrame
        cumulative produced/injected volumes (sm3) calculated from DATA. 
        If MAX_TS is provided => resampling is done so that maximal time step => MAX_TS
    '''

    def __init__(self,
                 data=None, cumulative=False,
                 date_format=None, start_date=None,
                 max_ts=None,
                 print_log=False, print_warnings=True
                 ):
        '''
        creates a Flow class instance

        Parameters
        ----------
        data : pd.DataFrame  
            production/injection data in a flexible format like: 
                dd.mm.yyyy     oil (sm3)   water (sm3)   gas (sm3) gas inj (sm3)  \n
            0    01.01.1957        0.0          0.0           0         0         \n
            1    01.02.1957        100          0.0           9       2000        \n
            2    01.03.1957        0.0          0.0           0         0         \n

            The following conventions are used:
            1. The first column must be time. Its format may be described in the 
            column name (see above) or by DATE_FORMAT (see description below)
            2. The first date corresponds to zero production and initial pressure
            3. Volumes may be provided as cumulative (by setting CUMULATIVE=True, see below) 
            or produced/injection volumes between two dates. I.e. in the example above 
            oil production (100) and injection (2000) at '01.02.1957' is assigned to 
            the interval between '01.01.1957' and '01.02.1957' 
            4. Names of volume columns should contain 'oil', 'gas', 'wat', 'co2' to distinguish 
            between fluids. Examples: 
            5. Injection columns are marked by 'inj' ('injection', 'inj', 'gas_inj'). 
            Columns without 'inj' in names are assumed to be production ones
            6. Names of volume columns are case insensitive

        date_format : string, optional 
            format for time column in DATA
            Default: the name of the first column of DATA
            The following formats are supported:
            * 'dd.mm.yyyy'/'yyyy.dd.mm'/'yyyy.mm.dd'/ etc.
            * 'day'/'days'
            * 'year'/'years'
            * 'datetime'
            * format codes from https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
                example: '%d-%b-%Y' => '01-JAN-2020'

        cumulative : bool, optional 
            are volumes in DATA cumulative?

        start_date: datetime.datetime, optional
            used along with 'day' or 'year' formats mostly for plotting.
            Defaults:
            * today if date_format=='days' or 'years'
            * else: the first date in DATA
        Example: datetime.datetime(2022,7,14)            

        max_ts : int, float, optional 
            maximal time step (days) which used to resample DATA into TOTALS and 
            RATES attributes and to control time step    

        print_log : bool
            print or not messages

        print_warnings: bool
            print or not warnings
            
        Returns
        -------
        rates : pd.DataFrame, attribute
            produced/injected rates(sm3/day) calculated from DATA. 
            If MAX_TS is provided => resampling is done so that maximal time step => MAX_TS

        totals : pd.DataFrame, attribute
            cumulative produced/injected volumes (sm3) calculated from DATA. 
            If MAX_TS is provided => resampling is done so that maximal time step => MAX_TS
        '''
        data = data.copy().reset_index(drop=True)
        self.cumulative = cumulative
        self.data = data.copy()
        self.max_ts = max_ts

        if not isinstance(data, pd.DataFrame):
            raise TypeError('DATA must be pd.DataFrame!')
        cols = data.columns
        if date_format == None:
            if print_log:
                print('message: DATE_FORMAT is not specified => checking the first column of DATA ...')
            if isinstance(data.iloc[0, 0], datetime.datetime):
                date_format = 'datetime'
            else:
                date_format = cols[0]

        if date_format.find('%') == -1:
            date_format = date_format.lower()  # just in case

        new_cols = {}
        for col in cols:
            new_cols[col] = col
            for fluid in ['oil', 'wat', 'gas', 'co2']:
                temp2 = ''
                if col.lower().find(fluid) > -1:
                    temp2 += fluid
                    if col.lower().find('inj') > -1:
                        temp2 += '_inj'
                    else:
                        temp2 += '_prod'
                    new_cols[col] = temp2
                    # just in case: converting to float
                    data.loc[:, col] = data.loc[:, col].astype(float)

                    if not cumulative:
                        data.loc[:, col] = data[col].cumsum()

        data.rename(columns=new_cols, inplace=True)

        del new_cols, temp2
        cols = data.columns

        if date_format.find('yyyy') > -1:
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            frmt = date_format.lower().replace('dd','%d').\
                    replace('mm','%m').replace('yyyy','%Y')
            data['date'] = pd.to_datetime(data['date'], format = frmt)
            data['days'] = (data['date'] - data.loc[0, 'date']).dt.days
            data['years'] = data['days']/365.25

        elif re.match('datetime', date_format):
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            data['days'] = (data['date'] - data.loc[0, 'date']).dt.days
            data['years'] = data['days']/365.25

        elif date_format.find('%') > -1:
            # https://docs.python.org/3/library/datetime.html#datetime.datetime.strptime
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            data['date'] = pd.to_datetime(data['date'], format = date_format)
            data['days'] = (data['date'] - data.loc[0, 'date']).dt.days
            data['years'] = data['days']/365.25

        elif re.match('day', date_format):
            data['days'] = data[cols[0]]
            if 'day' in cols: del data['day']
            if start_date == None:
                start_date = datetime.datetime.today()
            data['date'] = start_date + pd.to_timedelta(data['days'], unit='day')
            data['years'] = data['days']/365.25

        elif re.match('year', date_format):
            data['years'] = data[cols[0]]
            if 'year' in cols: del data['year']
            if start_date == None:
                start_date = datetime.datetime.today()
            data['days'] = data['years']*365.25
            data['date'] = start_date + pd.to_timedelta(data['days'], unit='day')
        else:
            if print_warnings:
                warn('\n   No processing is done! There may be an error. Check data and date_format specification!') 

        if start_date is None:
            start_date = data['date'][0]

        if max_ts is not None:
            df2 = pd.DataFrame(columns=data.columns, dtype='float64')
            df2.days = np.arange(0, data['days'].max(), max_ts)
            #data = data.append(df2)
            data = pd.concat([data, df2])
            data.drop_duplicates(subset='days', keep='first', inplace=True)

            data.sort_values(by='days', inplace=True)

            data.set_index('days', inplace=True)
            data = data.interpolate(method='index', fill_value=0)
            data.reset_index(drop=False, inplace=True)

            data['date'] = start_date + pd.to_timedelta(data['days'], unit='day')

        self.date_format = date_format
        self.start_date = start_date
        self.totals = data.copy()  # cumulative volumes

        # rates
        cols = data.columns
        dt = data['days'].diff().iloc[1:]
        for fluid in ['oil', 'wat', 'gas', 'co2']:
            for qwe in ['prod', 'inj']:
                col = fluid + '_' + qwe
                if col in cols:
                    dr = data.loc[:, col].diff().iloc[1:]
                    if (dr<0).any():
                        raise ValueError(f'Negative values in {fluid} {qwe}. rate! Check the inputs!')
                    data.loc[1:, col] = dr/dt
        self.rates = data  # rates


    def add_rows(self, t, use_rates=False, num_rows=1, \
        oil_prod=0, wat_prod=0, gas_prod=0, co2_prod=0, \
        oil_inj=0,  wat_inj=0,  gas_inj=0,  co2_inj=0,
        ):
        '''to add production/injection row(s) in the end of 

        Parameters
        ----------
        t : int, float, datetime.datetime, datetime.timedelta
            new time to add. The following modes are available :
            * int, float => number of days to add after the 
            * datetime.datetime => new date is added to the end
            * datetime.timedelta is added to the end

        use_rates : bool
            volumes/rates switch for fluid production/injection:
            * if True  => rates (sm3/day) will be used in oil_prod ... co2_inj
            * if False => volumes (sm3)
    
        num_rows : int
            number of time intervals to divide the period before t.
            The rates are kept equal, whereas the volumes are evenly split.

        oil_prod ... co2_inj : float, optional
            oil, water, gas, CO2 volumes (or rates)

        Returns
        -------        
        additional rows (and optionally columns) in totals and rates
        '''
        sec_in_day = 86400
        sec_in_year = 365.25*sec_in_day
        last_date = self.totals['date'].iat[-1] 
        last_day =self.totals['days'].iat[-1] 
        last_year=self.totals['years'].iat[-1]        

        clmns = self.totals.columns

        temp_dict0 = {}
        temp_totals = {}
        for fl in ['oil','wat','gas','co2']:
            for md in ['prod', 'inj']:
                nm = f'{fl}_{md}'
                vl = eval(nm) 
                if (vl > 0) or (nm in clmns): 
                    temp_dict0[nm] = vl
                    temp_totals[nm] = []
                if (vl > 0) and (nm not in clmns): 
                    self.rates[nm] = 0
                    self.totals[nm] = 0
        
        temp_totals['date'] = []
        temp_totals['days'] = []
        temp_totals['years'] = []                    

        if isinstance(t, datetime.datetime):
            new_date = t
            tdelta = (new_date - last_date)
            dt_sec = tdelta.total_seconds()
            dt_days = dt_sec/sec_in_day
            dt_years = dt_sec/sec_in_year 
        elif isinstance(t, (float, int)):
            dt_days = t
            dt_years = t/365.25
            tdelta = datetime.timedelta(days=dt_days)
            new_date = last_date + tdelta
        elif isinstance(t, datetime.timedelta):
            tdelta = t
            dt_sec = tdelta.total_seconds()
            dt_days = dt_sec/sec_in_day
            dt_years = dt_sec/sec_in_year 
            new_date = last_date + tdelta 
        else:
            raise ValueError('unknown t type')

        ln = np.linspace(0,1,num_rows+1)[1:]
        
        rates_df = pd.DataFrame()
        rates_df['date'] = last_date + tdelta*ln
        rates_df['years'] = last_year + dt_years*ln
        rates_df['days'] = last_day + dt_days*ln
        totals_df = rates_df.copy()

        if use_rates:
            for k in temp_dict0:
                totals_df[k] = temp_dict0[k]*dt_days*ln + self.totals[k].iloc[-1]
                rates_df[k] = temp_dict0[k]
        else:
            for k in temp_dict0:
                totals_df[k] = temp_dict0[k]*ln + self.totals[k].iloc[-1]
                rates_df[k] = temp_dict0[k]/dt_days 
        
        self.rates  = pd.concat([self.rates, rates_df]).fillna(0)
        self.totals = pd.concat([self.totals, totals_df]).fillna(0)
        self.rates.reset_index(drop=True, inplace=True)
        self.totals.reset_index(drop=True, inplace=True)


    def plot(self, fluids=None,
             oil_density=900,
             gas_density=0.75,
             wat_density=1000,
             co2_density=1.8472,
             date=False, rate_mode='mass',
             ax=None):
        '''quick plot of flow rates

        Parameters
        ----------
        fluids : instance of Fluids class, optional
            if provided => the fluid densities will be retrieved from it and
            other densities will be superseded 

        oil_density : int, float, optional
            oil density @SC (kg/sm3)
            
        wat_density : int, float, optional
            water density @SC (kg/sm3)

        gas_density : int, float, optional
            gas density @SC (kg/sm3)

        co2_density : int, float, optional
            co2 density @SC (kg/sm3)

        date : bool, optional
            if True => xaxis='date' else 'days'
        
        rate_mode : str ('mass'|'SC'), optional, case insensitive
            if 'mass' => mass rate
            if 'SC' => volumetric rate in standard conditions

        ax : matplotlib.axes, optional
            axes to use for plotting

        Returns
        -------
        axes of production data plot
        '''
        rate_mode = rate_mode.lower()
        if rate_mode == 'mass':
            if fluids is None:
                dens_dict = {}
                dens_dict['oil'] = oil_density
                dens_dict['gas'] = gas_density
                dens_dict['water'] = wat_density
                dens_dict['co2'] = co2_density
            else:
                if fluids.oil is not None:
                    dens_dict['oil'] = fluids.oil.density
                if fluids.water is not None:
                    dens_dict['water'] = fluids.water.density
                if fluids.gas is not None:
                    dens_dict['gas'] = fluids.gas.density
                if fluids.co2 is not None:
                    dens_dict['co2'] = fluids.co2.density

        if ax is None:
            ax = plt.gca()

        ns = self.rates.shape[0]
        clrs_dict = {'oil': 'tab:green',
                     'water': 'tab:blue',
                     'gas': 'tab:red',
                     'co2': 'darkviolet',
                     'aquifer': 'tab:cyan'}

        cols = self.rates.columns
        if date:
            # x = np.repeat(self.rates['date'].T.values,2)[:-1]
            x = self.rates['date'].values
        else:
            # x = np.repeat(self.rates['days'].T.values,2)[:-1]
            x = self.rates['days'].values

        # production
        y1 = np.zeros(ns,)
        lbls = []
        for fluid in ['oil', 'water', 'gas', 'co2']:
            
            col = fluid[:3] + '_prod'
            if rate_mode == 'mass':
                rho = dens_dict[fluid]/1000
            if rate_mode == 'sc':
                rho = 1

            if col in cols and ((self.rates[col] == 0).all() == False):
                lbls.append(fluid)
                y2 = self.rates[col].values.T
                #y2 = rho*np.repeat(y2,2)[1:] + y1
                y2 = rho*y2 + y1

                ax.fill_between(x, y1, y2,
                                color=clrs_dict[fluid],
                                label=fluid,
                                linewidth=0.1, step='post')
                y1 = y2

        # injection
        y1 = np.zeros(ns,)
        for fluid in ['oil', 'water', 'gas', 'co2']:
            col = fluid[:3] + '_inj'
            if rate_mode == 'mass':
                rho = dens_dict[fluid]/1000
            if rate_mode == 'sc':
                rho = 1

            if col in cols and ((self.rates[col] == 0).all() == False):
                #y2 = -self.rates[col].values.T
                #y2 = rho*np.repeat(y2,2)[1:] + y1
                y2 = -self.rates[col].values.T
                y2 = rho*y2 - y1
                if fluid not in lbls:
                    ax.fill_between(x, y1, y2,
                                    color=clrs_dict[fluid],
                                    label=fluid, linewidth=0.1,
                                    step='post')
                else:
                    ax.fill_between(x, y1, y2,
                                    color=clrs_dict[fluid],
                                    linewidth=0.1,
                                    step='post')
                y1 = y2

        plt.grid()
        plt.legend()
        if date == False:
            plt.xlabel('day')

        if rate_mode == 'mass':
            ax.set_ylabel('t/day')
            ax.set_title('production/injection mass rate')
        if rate_mode == 'sc':
            ax.set_ylabel('sm3/day')
            ax.set_title('production/injection rate (SC)')

        return ax

class PressureMeasurements(__Au__):
    '''historical pressure measurements (mostly for visualization purposes)'''

    def __init__(self, data=None, date_format=None, start_date=None):
        '''Parameters
        ----------
        data : pd.DataFrame
            historical pressure measurements
            1st column (0th) must contain time.
            Its name is specified in DATE_FORMAT below
            Other columns can include (only 'value' is obligatory):
            * 'value': float
                measurement value (bar)
            * 'type': str
                'static' (default) or 'dynamic' 
            * 'estimate':  str
                use to indicate if pressure was estimated:
                - 'max' - maximal estimate
                - 'min' - minimum estimate
                - 'mean', or 'True'/True, or 'yes' - mean estimate
                - 'False'/False or 'no', or anything else - measured pressure
                by default (for all other identifiers) all measurements are direct
                
            * 'well': str, optional
                well name. used for tooltips 
            * 'comment': str, optional     

        date_format : str, optional 
            format for time column in DATA
            Default: the name of the first column of DATA
            The following formats are supported:
            * 'dd.mm.yyyy'/'yyyy.dd.mm'/'yyyy.mm.dd'/ etc.
            * 'day'/'days'
            * 'year'/'years'
            * 'datetime'
            * format codes from https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
            example: '%d-%b-%Y' => '01-JAN-2020'

        start_date: datetime.datetime object, optional
            used along with 'day' or 'year' formats mostly for plotting.
            Defaults:
            * today if date_format=='days' or 'years'
            * else: the first date in DATA     
            Example: datetime.datetime(2022,7,14)
        '''
        cols = data.columns
        if 'type' not in cols: data['type']='static'
        if 'well' not in cols: data['well']=None
        # just in case for backward compatibility
        if 'mean' in cols: data = data.rename(columns={'mean': 'value'})
        
        if 'estimate' not in cols: 
            data['estimate'] = False
        else:
            # go through the column to ensure the cont is ok 
            for i in data.index:
                foo = data.loc[i,'estimate']
                if (foo is None):
                    data.loc[i,'estimate'] = False
                elif isinstance(foo, str):
                    foo = foo.lower()
                    if foo[:3] == 'min':
                        data.loc[i,'estimate'] = 'min'
                    elif foo[:3] == 'max':
                        data.loc[i,'estimate'] = 'max'
                    elif (foo[:4]=='mean') | (foo[:3]=='yes') | (foo[:5]=='true'):
                        data.loc[i,'estimate'] = True                           
                    else:
                        data.loc[i,'estimate'] = False
                elif isinstance(foo, bool):
                    pass
                else:
                    data.loc[i,'estimate'] = False

        if 'comment' not in cols:  data['comment'] = None
        
        # creating format
        if date_format == None:
            if isinstance(data.iloc[0, 0], datetime.datetime):
                date_format = 'datetime'
            else:
                date_format = cols[0]

        if start_date is None:
            start_date = datetime.datetime.today()

        if date_format.find('%') == -1:
            date_format = date_format.lower()  # just in case

        if date_format.find('yyyy') > 0:
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            frmt = date_format.lower().replace('dd','%d').\
                      replace('mm','%m').replace('yyyy','%Y')
            data['date'] = pd.to_datetime(data['date'], format = frmt)
            data['days'] = (data['date'] - start_date).dt.days
            data['years'] = data['days']/365.25
            
        elif re.match('datetime', date_format):
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            data['days'] = (data['date'] - start_date).dt.days
            data['years'] = data['days']/365.25

        elif date_format.find('%') > -1:
            # https://docs.python.org/3/library/datetime.html#datetime.datetime.strptime
            data.rename(columns={data.columns[0]: 'date'}, inplace=True)
            frmt = date_format.lower().replace('dd','%d').\
                    replace('mm','%m').replace('yyyy','%Y')
            data['date'] = pd.to_datetime(data['date'], format = frmt)
            data['days'] = (data['date'] - start_date).dt.days
            data['years'] = data['days']/365.25

        elif re.match('day', date_format):
            data['days'] = data[cols[0]]
            if 'day' in cols: del data['day']
            data['date'] = start_date + pd.to_timedelta(data['days'], unit='day')
            data['years'] = data['days']/365.25

        elif re.match('year', date_format):
            data['years'] = data[cols[0]]
            if 'year' in cols:
                del data['year']
            data['days'] = data['years']*365.25
            data['date'] = start_date + pd.to_timedelta(data['days'], unit='day')

        self.data = data

class Fluids(__Au__):
    '''set of PVT properties for oil, water, gas and CO2
    
    Subclasses
    ----------
    * Oil
    * Water
    * Gas
    * CO2
    '''

    def __init__(self, oil=None, water=None, gas=None, co2=None):
        '''
        Parameters
        ---------- 
        oil : Fluid.Oil, optional
            PVT properties of live oil
        water : Fluid.Water, optional
            PVT properties of water
        gas : Fluid.Gas, optional
            PVT properties of free gas
        co2 : Fluid.CO2
            PVT properties of CO2
        '''
        self.oil = oil
        self.water = water
        self.gas = gas
        self.co2 = co2

    def use_default_PVT(self):
        '''use to set up a PVT instance with default properties'''
        self.oil = Fluids.Oil()
        self.water = Fluids.Water()
        self.gas = Fluids.Gas()
        self.co2 = Fluids.CO2()
        return self

    class Water(__Au__):
        '''water PVT properties

        Attributes
        ----------
        cw : float
            compressibility (1/bar)
        density : float
            density in SC (kg/sm3)
        fvf_ref: float, optional
            reference formation vol. factor (rm3/sm3) at ...
        p_ref : float 
            reference pressure (bar)
        '''
        def __init__(self, cw=4e-5, density=1000, fvf_ref=1, p_ref=1):
            self.cw = cw
            self.density = density
            self.fvf_ref = fvf_ref
            self.p_ref = p_ref

    class Oil(__Au__):
        '''live oil PVT properties'''

        def __init__(self,
                     fvf_table=fvf_oil_lbr1,
                     rs_table=rs_of_pb_lbr1,
                     co_table=[[1, 1e-4], [2000, 1e-4]],
                     density=900,
                     dvf=1, dtc=1, drsdt=np.inf):
            '''
            Attributes
            ----------
            fvf_table : pd.DataFrame, ndarray
                table of oil formation volume factor (FVF) vs. bubble point pressure:
                * 1st column: pressure (bar)
                * 2nd column: FVF (sm3/sm3)

            rs_table : pd.DataFrame, ndarray  
                table of gas solution ratio vs. bubble point pressure:
                * 1st column: pressure (bar)
                * 2nd column: Rs (sm3/sm3)

            co_table : pd.DataFrame, ndarray
                table of oil compressibility vs. bubble point pressure:   
                * 1st column: pressure (bar)   
                * 2nd column: oil compressibility (1/bar)   

            density : float, int
                density in SC (kg/sm3)

            drsdt : float, int, optional
                d(Rs)/dt, max. increase rate of solution GOR (sm3/sm3/day)   
                (analogue of DRSDT in e100) 

            dvf : float, int, optional
                fraction of free GIIP that can be dissolved in oil (fraction)   

            dtc : float, optional
                dissolution tapering coefficient which limits dissolution of   
                gas in oil within a time step (fraction)   
                1 (default) -> dissolution without tapering, 0 -> no dissolution   
            '''                     
            self.density = density
            self.fvf_table = fvf_table
            self.rs_table = rs_table
            self.co_table = co_table
            self.drsdt = drsdt
            self.dvf = dvf
            self.dtc = dtc            

            if not isinstance(self.fvf_table, pd.DataFrame):
                self.fvf_table = \
                    pd.DataFrame(data=self.fvf_table,
                                 columns=['p_bub (bar)', 'FVF (rm3/sm3)'])

            if not isinstance(self.co_table, pd.DataFrame):
                self.co_table = \
                    pd.DataFrame(data=self.co_table,
                                 columns=['p_bub (bar)', 'compressibility (1/bar)'])

            if not isinstance(self.rs_table, pd.DataFrame):
                self.rs_table = \
                    pd.DataFrame(data=self.rs_table,
                                 columns=['p_bub (bar)', 'Rs (sm3/sm3)'])

    class Gas(__Au__):
        '''gas PVT properties:
        
        Attributes
        ----------
        fvf_table : pd.DataFrame, ndarray
            table of gas formation volume (FVF) factor vs. pressure:
            * 1st column: pressure (bar)
            * 2nd column: FVF (sm3/sm3)            
        density : float, int, optional
            density in SC (kg/sm3)
        '''

        def __init__(self, fvf_table=fvf_gas_lbr1, density=0.75):

            self.fvf_table = fvf_table
            self.density = density

            if not isinstance(self.fvf_table, pd.DataFrame):
                self.fvf_table = \
                    pd.DataFrame(data=self.fvf_table,
                                 columns=['p (bar)', 'FVF (rm3/sm3)'])

    class CO2(__Au__):
        '''CO2 PVT properties:
        
        Parameters
        ----------
        fvf_table : pd.DataFrame, ndarray
            table of gas formation volume factor (FVF) vs. pressure:
            * 1st column: pressure (bar)
            * 2nd column: FVF (sm3/sm3)  
        density : float, int, optional
            density in SC (kg/sm3)       
        '''

        def __init__(self, fvf_table=None, density=1.8472):
            self.fvf_table = fvf_table
            self.density = density
            if not isinstance(self.fvf_table, pd.DataFrame):
                self.fvf_table = \
                    pd.DataFrame(data=self.fvf_table,
                                 columns=['p (bar)', 'FVF (rm3/sm3)'])


class Aquifer:
    '''Aquifer models
    
    Subclasses:
    ----------    
    * Fetkovich
    * SteadyState
    '''

    def __init__(self):
        pass

    # available models: Fetkovich, Fetkovich_radial, Fetkovich_linear, steady_state, Carter-Tracey
    class Fetkovich(__Au__):
        '''Fetkovich aquifer with volume V (sm3) and productivity index PI (sm3/day/bar)'''

        def __init__(self, v=0, pi=0, ct=None, p0=None, mode='original'):
            '''
            Parameters
            ----------
            v : float
                aquifer volume (sm3)
            pi : float
                aquifer productivity index (sm3/day/bar)             
            ct : None,float
                total aquifer compressibility (1/bar) that overrides the default 
                value calculated as:
                ct = cf + cw = Reservoir.cf + Fluids.water.cw
            p0 : float
                initial aquifer pressure (bar)
                default: the initial reservoir pressure (case.reservoir.p0)
            mode : str, optional
                the following slightly different formulations are available:
                * 'original' uses the formulation from the original (SPE-2603-PA, [1]) paper:
                    p_av_ = 0.5*(p[i-1] + p_)  # average reservoir pressure
                    we_ = (p_aq[i-1] - p_av_)*Ct_Vw0*(1 - np.exp(-pi_Ct_Vw0*dt))                
                
                * 'modified' - yields the best match with a commercial simulator
                  (refer 'example_radial_aquifer'):
                    we_ = (p_aq[i-1] - p_)*Ct_Vw0*(1 - np.exp(-pi_Ct_Vw0*dt))
                
                
                where:
                - Ct_Vw0 = v*(cw+cf)
                - pi_Ct_Vw0 = pi/Ct_Vw0

                Ref.:  
                1.  Fetkovich, M.J.. "A Simplified Approach to Water Influx Calculations-Finite Aquifer Systems."
                    J Pet Technol 23 (1971): 814–828. doi: https://doi.org/10.2118/2603-PA
            '''
            self.model = 'Fetkovich'

            self.v = v
            self.pi = pi
            self.ct = ct
            self.mode = mode
            self.p0 = p0            

    class SteadyState(__Au__):
        '''Steady state infinite aquifer with productivity index PI (sm3/bar/day) '''
        def __init__(self, pi=0):
            self.model = 'SteadyState'
            self.pi = pi

class Case(__Au__):
    '''all data for material balance model combined
    
    Methods:
    --------
    run : run material balance code  

    usc_run : estimate Ultimate Storage Capacity - maximal amount 
    of CO~2~ that can be accommodated in the reservoir at given pressures 
    and other conditions specified in case.usc
    '''

    def __init__(self,
                 name=None,
                 description=None,
                 reservoir=None,
                 fluids=None,
                 flows=None,
                 pressure_measurements=None,
                 aquifer=None,
                 usc=None,
                 results=None):
        '''
        Parameters
        ----------        
        name : str, optional
            case name
        description : str, optional
            case description 
        reservoir : unicellar.Reservoir
            reservoir properties
        fluids : unicellar.Fluids
            PVT properties
        flows : unicellar.Flows
            fluid flows in and out of the reservoir. Optional for USC runs
        pressure_measurements : unicellar.PressureMeasurements, optional
            historical pressure measurements
        aquifer : unicellar.Aquifer, optional
            aquifer model
        usc : pd.DataFrame
            scenarios to estimate Ultimate Storage Capacity (USC). 
            Columns (use usc_template to get a prefilled template):
            * p_max - maximal pressure (bar)
            * oil_prod - oil production (sm3)
            * wat_prod - water production (sm3)
            * gas_prod - gas production (sm3)
            * co2_prod - CO2 production (sm3)
            * oil_inj - oil injection (sm3)
            * wat_inj - water injection (sm3)
            * gas_inj - gas injection (sm3)
            * co2_inj - CO2 injection (sm3)
        '''
        self.name = name
        self.description = description
        self.reservoir = reservoir
        self.fluids = fluids
        self.flows = flows
        self.pressure_measurements = pressure_measurements
        self.aquifer = aquifer
        self.results = results
        self.usc = usc

    def run(self, jmax=10, max_error=1e-12, \
            print_log=False, print_warnings=True, request_qQ=False):
        '''run the material balance code 
        
        Parameters:
        ----------
        self : unicellar.Case
            material balance model
        print_log : bool, optional
            print log or not
        print_warnings : bool, optional
            print warnings or not
        jmax : int, optional
            maximum number of iterations per step
        max_error : float, optional
            target material balance error (fraction of PV) for any step
        request_qQ : bool, optional
            add produced/injected rates and totals to self.results dataframe
        
        Returns:
        --------    
        self.results : pd.DataFrame
            simulation results
        '''
        self = run(self, jmax, max_error, print_log, print_warnings, request_qQ)

    def usc_run(self, print_log=False, print_warnings=True):
        '''evaluate of Ultimate Storage Capacity (USC)
        
        Parameters:
        ----------
        self : unicellar.Case
            material balance model with USC scenarios in case.USC
        print_log : bool, optional
            print log or not
        print_warnings : bool, optional
            print warnings or not                

        Returns
        --------
        self.usc : pd.DataFrame
            columns with storage capacities are added: 'USC (rm3)', 'USC (sm3)', 'USC (t)'
        '''
        self = usc_run(self, print_log, print_warnings)
        # return self

    def plot_vector(self, x='t', y='p', ax=None, 
                    color=None, linestyle='-',
                    label='model', title=None, ylabel=None, ylim=None):
        '''plot result vectors in Matplotlib.

        Parameters:
        ----------
        self : unicellar.Case
            material balance model

        x, y : str     
            Vectors to plot. The following mnemonics may be used:
            * 't': 'days'   \n
            * 'p': 'pressure (bar)'   \n
            * 'p_aq': 'aquifer pressure (bar)'   \n
            * 'paq': 'aquifer pressure (bar)'   \n
            * 'aaqt': 'cumulative aquifer influx (sm3)'   \n
            * 'faqt': 'cumulative aquifer influx (sm3)'   \n
            * 'aaqr': 'aquifer influx rate (sm3/day)'   \n
            * 'faqr': 'aquifer influx rate (sm3/day)'
            * 'nwrrc': 'net withdrawal rate (rm3)'    \n

        ax : matplotlib.axes, optional
            axes to use for plotting

        color, label, linestyle, title, ylabel, ylim : totaly optional 
            matplotlib parameters to style the plot

        Returns
        -------
        matplotlib.axes
        '''
        
        mnemonics = {
            't': 'days',
            'p': 'pressure (bar)',
            'p_aq': 'aquifer pressure (bar)',
            'paq': 'aquifer pressure (bar)',
            'aaqt': 'cumulative aquifer influx (sm3)',
            'faqt': 'cumulative aquifer influx (sm3)',
            'aaqr': 'aquifer influx rate (sm3/day)',
            'faqr': 'aquifer influx rate (sm3/day)',
            'nwrrc': 'net withdrawal rate (rm3/day)'}

        if self.results is None:
            print('no results found')
        else:
            cols = self.results.columns
            if ax is None:
                ax = plt.gca()
                # if plt.gca().lines == []
                #    plt.figure(figsize=(16,9), dpi=300)

            # looking for x
            if mnemonics.get(x, None) != None:
                x = mnemonics.get(x, None)
            else:
                for col in cols:
                    if re.match(x, col):
                        x = col
                        break
            # looking for y
            if mnemonics.get(y, None) != None:
                y = mnemonics.get(y, None)
            else:
                for col in cols:
                    if re.match(y, col):
                        y = col
                        break

            xx = self.results[x]
            yy = self.results[y]

            ax.plot(xx, yy, linestyle, label=label, color=color)
            if y == 'pressure (bar)' and (x in ['days', 'years', 'date']) and \
                    self.pressure_measurements is not None:
                x2 = self.pressure_measurements.data[x]
                clrs = ['dimgrey', 'dimgrey', 'dimgrey']
                what = ['value', 'min', 'max']
                #what = ['value']
                for m, q, clr in zip(['o', 'v', '^'], what, clrs):
                    if q in self.pressure_measurements.data.columns:
                        y2 = self.pressure_measurements.data[q]
                        ax.plot(x2, y2, m, label='history', mew=2,
                                markerfacecolor='None', color=clr)
                plt.legend()

            if ylabel is not None:
                plt.ylabel(ylabel)
            else:
                plt.ylabel(y.split()[-1][1:-1])

            if title is not None:
                plt.title(title)

            if ylim is not None:
                plt.ylim(ylim)

            if x == 'date':
                plt.xticks(rotation=90)
            else:
                plt.xlabel(x)

            plt.tight_layout()
            plt.grid()

            return ax

    def plot_pv(self, x='date', ax=None, fig=False, pie=False):
        '''plot fluid volumes @RC in Matplotlib

         Parameters
        ----------    
        self : unicellar.Case
            material balance model

        if self.results == None:
            plots a pie chart of initial in-place volumes (RC)
        else:
        plots an area chart of in-place volumes (RC) vs. time
        x - 'days', 'years' or 'date'
        fig - if True and ax=None, creates a figure
        pie - show initial volumes
        '''

        p0 = self.reservoir.p0
        oil = (True if self.fluids.oil is not None else False)
        gas = (True if self.fluids.gas is not None else False)
        water = (True if self.fluids.water is not None else False)
        co2 = (True if self.fluids.co2 is not None else False)

        clrs_dict = {
            'oil': 'tab:green',
            'water': 'tab:blue',
            'gas': 'tab:red',
            'co2': 'darkviolet'}

        pv = []
        labels = []
        if self.results is None or pie == True:
            if ax is None:
                if fig:
                    plt.figure(figsize=(7, 7), dpi=300)
                ax = plt.gca()
            clrs = []
            print('Fluid volumes in RC:')
            if water:
                cw = self.fluids.water.cw

                fvf_ref = self.fluids.water.fvf_ref
                p_ref = self.fluids.water.p_ref
                x = cw*(p0 - p_ref)
                bw0 = fvf_ref/(1+x+0.5*x**2)

                wip0 = self.reservoir.wiip
                foo = wip0*bw0
                if foo > 0:
                    pv.append(foo)
                    labels.append('water')
                    clrs.append(clrs_dict['water'])
                    print(f'   water:  {foo:.3e} rm3')

            if co2:
                x = self.fluids.co2.fvf_table.iloc[:, 0]
                y = self.fluids.co2.fvf_table.iloc[:, 1]
                rep_fvf_co2 = interp1d(x, 1/y, fill_value="extrapolate")
                def fvf_co2(x): return 1/rep_fvf_co2(x)
                co2_den_sc = self.fluids.co2.density
                del x, y
                co2ip0 = self.reservoir.co2iip
                bco20 = fvf_co2(p0)
                foo = co2ip0*bco20
                if foo > 0:
                    pv.append(foo)
                    labels.append('CO2')
                    clrs.append(clrs_dict['co2'])
                    print(f'   CO2:    {foo:.3e} rm3')

            if gas:
                x = self.fluids.gas.fvf_table.iloc[:, 0]
                y = self.fluids.gas.fvf_table.iloc[:, 1]
                rep_fvf_gas = interp1d(x, 1/y, fill_value="extrapolate")
                def fvf_gas(x): return 1/rep_fvf_gas(x)
                del x, y

                gip0 = self.reservoir.giip
                gipf0 = self.reservoir.giip  # is corrected below
                bg0 = fvf_gas(p0)
                if oil:
                    # FVF vs. p_bub
                    x = self.fluids.oil.fvf_table.iloc[:, 0]  # pb
                    y = self.fluids.oil.fvf_table.iloc[:, 1]  # FVF
                    fvf_oil = interp1d(x, y, kind='linear',
                                       fill_value="extrapolate")
                    # oil compressibility vs. p_bub
                    x = self.fluids.oil.co_table.iloc[:, 0]
                    y = self.fluids.oil.co_table.iloc[:, 1]
                    co_of_pb = interp1d(x, y, fill_value="extrapolate")
                    del x, y
                    pb = self.fluids.oil.rs_table.iloc[:, 0]
                    rs = self.fluids.oil.rs_table.iloc[:, 1]
                    pb_of_rs = interp1d(
                        rs, pb, kind='linear', fill_value="extrapolate")
                    rs_of_pb = interp1d(
                        pb, rs, kind='linear', fill_value="extrapolate")
                    del rs, pb
                    oip0 = self.reservoir.stoiip
                    if self.reservoir.rs0 is None:
                        self.reservoir.rs0 = rs_of_pb(p0)
                        # print( 'rs0 is not specified')
                        # print(f ' => rs0 estimated from rs_table: {self.reservoir.rs0:.2f} sm3/sm3')
                        print(
                            f'rs0 is not specified => estimated from rs_table: {self.reservoir.rs0:.2f} sm3/sm3')
                    rs0 = self.reservoir.rs0
                    pb0 = pb_of_rs(rs0)
                    bo0 = fvf_oil(pb0)*(1 - (p0 - pb0)*co_of_pb(pb0))
                    gips0 = oip0*rs0
                    gipf0 -= gips0
                    if gipf0 < 0:
                        raise ValueError(
                            'gipf0<0 free gas is negative /n check rs0 and giip')
                    foo = oip0*bo0
                    if foo > 0:
                        pv.append(foo)
                        labels.append('oil')
                        clrs.append(clrs_dict['oil'])
                        print(f'   oil:    {foo:.3e} rm3')

                foo = gipf0*bg0
                if foo > 0:
                    pv.append(foo)
                    labels.append('gas')
                    clrs.append(clrs_dict['gas'])
                    print(f'   gas:    {foo:.3e} rm3')

            ax.pie(pv, labels=labels,
                   autopct='%1.1f%%',
                   pctdistance=0.9, colors=clrs,
                   shadow=False, startangle=90)
            # ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Initial fluid in-place volumes in RC')
            # plt.show()

        else:

            if ax is None:
                if fig:
                    plt.figure(figsize=(9, 6), dpi=300)
                ax = plt.gca()

            oil = (True if self.fluids.oil is not None else False)
            gas = (True if self.fluids.gas is not None else False)
            water = (True if self.fluids.water is not None else False)
            co2 = (True if self.fluids.co2 is not None else False)

            xx = self.results[x].values
            y1 = np.zeros((len(xx),))

            if oil:
                y2 = y1 + \
                    self.results['OIP (sm3)'].values * \
                    self.results['oil FVF (rm3/sm3)'].values
                ax.fill_between(xx, y1, y2, color=clrs_dict['oil'],
                                label='oil', linewidth=0.1, step='post')
                y1 = y2

            if gas:
                y2 = y1 + \
                    self.results['GIP (sm3)'].values * \
                    self.results['gas FVF (rm3/sm3)'].values
                ax.fill_between(xx, y1, y2, color=clrs_dict['gas'],
                                label='gas', linewidth=0.1, step='post')
                y1 = y2

            if water:
                y2 = y1 + \
                    self.results['WIP (sm3)'].values * \
                    self.results['water FVF (rm3/sm3)'].values
                ax.fill_between(xx, y1, y2, color=clrs_dict['water'],
                                label='water', linewidth=0.1, step='post')
                y1 = y2

            if co2:
                y2 = y1 + \
                    self.results['CO2IP (sm3)'].values * \
                    self.results['CO2 FVF (rm3/sm3)'].values
                ax.fill_between(xx, y1, y2, color=clrs_dict['co2'],
                                label='CO2', linewidth=0.1, step='post')
                y1 = y2

            ax.plot(xx[[0, -1]], y2[[0, 0]], '--', color='black')

            plt.title('Fluid in-place volumes in RC')
            if x == 'date':
                plt.xticks(rotation=90)
            else:
                plt.xlabel(x)

            plt.ylabel('rm3')
            plt.legend()
            plt.grid()
            plt.tight_layout()

        return ax

# %% functions
def run(case, jmax=10, max_error=1e-12, \
        print_log=True, print_warnings=True, request_qQ=False):
    '''run the material balance code 
    
    Parameters:
    ----------
    case : unicellar.Case
        material balance model
    print_log : bool, optional
        print log or not 
    jmax : int, optional
        maximum number of iterations per step
    max_error : float, optional
        target material balance error (fraction of PV) for any step
    request_qQ : bool, optional
        add produced/injected rates and totals to case.results dataframe
    stick_to_rs0 : bool, optional
        this option prevents free gas dissolution in the beginning if p_bub < p_res.
        The idea is to mimic behavior of commercial reservoir simulators where rs0 under gas cap
        can be set lower than the equilibrium value rs(p_res).
        For instance, in Eclipse it can be done by PBVD and RSVD keywords.
        If set to True, this option invokes the following behavior:
        1. as long as p_bub < p_res, gas dissolution is not allowed
        2. if rs becomes less rs0, then gas dissolution is allowed for all 
        subsequent timesteps        

    Returns:
    --------    
    case.results : pd.DataFrame
        simulation results
        '''
    num_steps = len(case.flows.totals.index)
    t = case.flows.rates.days
    cf = case.reservoir.cf
    stick_to_rs0 = case.reservoir.stick_to_rs0

    oil = (True if case.fluids.oil is not None else False)
    gas = (True if case.fluids.gas is not None else False)
    water = (True if case.fluids.water is not None else False)
    co2 = (True if case.fluids.co2 is not None else False)
    # initialization/ preparing interpolation functions for PVT
    p = np.zeros((num_steps,))  # pressure
    p[0] = case.reservoir.p0
    # preallocating oil, water, gas, CO2 production and injection
    op = np.zeros((num_steps,))
    wp = np.zeros((num_steps,))
    gp = np.zeros((num_steps,))
    co2p = np.zeros((num_steps,))
    oi = np.zeros((num_steps,))
    wi = np.zeros((num_steps,))
    gi = np.zeros((num_steps,))
    co2i = np.zeros((num_steps,))
    mb_error = {}
    if 'oil_prod' in case.flows.totals.columns:
        op[1:] = case.flows.totals['oil_prod'].diff()[1:]
    if 'wat_prod' in case.flows.totals.columns:
        wp[1:] = case.flows.totals['wat_prod'].diff()[1:]
    if 'gas_prod' in case.flows.totals.columns:
        gp[1:] = case.flows.totals['gas_prod'].diff()[1:]
    if 'co2_prod' in case.flows.totals.columns:
        co2p[1:] = case.flows.totals['co2_prod'].diff()[1:]
    if 'oil_inj' in case.flows.totals.columns:
        oi[1:] = case.flows.totals['oil_inj'].diff()[1:]
    if 'wat_inj' in case.flows.totals.columns:
        wi[1:] = case.flows.totals['wat_inj'].diff()[1:]
    if 'gas_inj' in case.flows.totals.columns:
        gi[1:] = case.flows.totals['gas_inj'].diff()[1:]
    if 'co2_inj' in case.flows.totals.columns:
        co2i[1:] = case.flows.totals['co2_inj'].diff()[1:]

    compaction_model = case.reservoir.compaction_model
    if isinstance(compaction_model, str):
        compaction_model = compaction_model.lower()
        if compaction_model == 'cf':
            pv_mult_func = lambda dp: np.exp(cf*dp)
        elif compaction_model == 'pv table' or compaction_model == 'pv_table':
            pv_table = case.reservoir.pv_table.copy()
            if isinstance(case.reservoir.pv_table, pd.DataFrame):
                pv_table = pv_table.to_numpy()
            dp = pv_table[:, 0]
            pv_mult = pv_table[:, 1]
            pv_mult_func = interp1d(dp, pv_mult, kind='linear')
        else:
            raise ValueError(
                f'compaction model "{compaction_model}" is unknown!')
    elif callable(compaction_model):
        pv_mult_func = compaction_model
    else:
        raise ValueError(f'compaction model is in a wrong format!')

    # net withdrawal
    no = op - oi  # oil
    nw = wp - wi  # water
    ng = gp - gi  # gas
    nco2 = co2p - co2i  # CO2
    # water
    pv = np.zeros((num_steps,))  # pore volume
    nwrrc = np.zeros((num_steps,))  # net withdrawal rate in RC
    if water:
        cw = case.fluids.water.cw
        bw_ref = case.fluids.water.fvf_ref
        pw_ref = case.fluids.water.p_ref        
        fvf_wat = lambda p: bw_ref*np.exp(cw*(pw_ref - p))
        wip = np.zeros((num_steps,))  # water-in-place
        wip[0] = case.reservoir.wiip
        bw = np.zeros((num_steps,))  # preallocation
        bw[0] = fvf_wat(p[0])
        pv[0] += wip[0]*bw[0]

    if co2:
        x = case.fluids.co2.fvf_table.iloc[:, 0]
        y = case.fluids.co2.fvf_table.iloc[:, 1]
        #fvf_co2 = interp1d(x,y)
        rep_fvf_co2 = interp1d(x, 1/y, fill_value="extrapolate")
        def fvf_co2(x): return 1/rep_fvf_co2(x)
        del x, y
        co2ip = np.zeros((num_steps,))  # CO2-in-place
        co2ip[0] = case.reservoir.co2iip
        bco2 = np.zeros((num_steps,))  # preallocation
        bco2[0] = fvf_co2(p[0])
        pv[0] += co2ip[0]*bco2[0]
        #mb_error['co2'] = np.zeros((num_steps,))

    if gas:
        x = case.fluids.gas.fvf_table.iloc[:, 0]
        y = case.fluids.gas.fvf_table.iloc[:, 1]
        rep_fvf_gas = interp1d(x, 1/y, fill_value="extrapolate")
        def fvf_gas(x): return 1/rep_fvf_gas(x)
        del x, y

        gip = np.zeros((num_steps,))  # gas-in-place
        gipf = np.zeros((num_steps,))  # free gas-in-place
        gips = np.zeros((num_steps,))  # gas in solution
        bg = np.zeros((num_steps,))  # preallocation of gas FVF
        bg[0] = fvf_gas(p[0])        
        
        giip = case.reservoir.giip
        fgiip = case.reservoir.fgiip
        if giip>0 and fgiip>0:
            ValueError('GIIP>0 and FGIIP>0. Only one must be defined!')   

        # oil should always contain gas
        if oil:
            bo = np.zeros((num_steps,))  # preallocation of oil FVF
            oip = np.zeros((num_steps,))  # oil-in-place preallocation            
            rs = np.zeros((num_steps,))  # GOR
            pb = np.zeros((num_steps,))  # bubble point pressure

            # FVF vs. p_bub
            x = case.fluids.oil.fvf_table.iloc[:, 0]  # pb
            y = case.fluids.oil.fvf_table.iloc[:, 1]  # FVF
            # fvf_oil = interp1d(x, y, kind='linear', fill_value="extrapolate")
            rep_fvf_oil = interp1d(x, 1/y, fill_value="extrapolate")
            def fvf_oil(x): return 1/rep_fvf_oil(x)
            # oil compressibility vs. p_bub
            x = case.fluids.oil.co_table.iloc[:, 0]
            y = case.fluids.oil.co_table.iloc[:, 1]
            co_of_pb = interp1d(x, y, fill_value="extrapolate")
            del x, y
            # pb_of_rs(rs) and rs_of_pb(pb)
            x = case.fluids.oil.rs_table.iloc[:, 0] # pb
            y = case.fluids.oil.rs_table.iloc[:, 1]        # rs
            pb_of_rs = interp1d(y, x, kind='linear', fill_value="extrapolate")
            rs_of_pb = interp1d(x, y, kind='linear', fill_value="extrapolate")
            del x, y            

            stoiip = case.reservoir.stoiip
            # estimate rs0 if needed            
            if case.reservoir.rs0 is None:
                rs0 = rs_of_pb(p[0])
                if print_log:
                    print(f'rs0 is not specified => estimated from rs_table: {rs0:.2f} sm3/sm3')
            else:
                rs0 = case.reservoir.rs0
                       
            sgiip = rs0*stoiip  
            if giip==0 and fgiip==0:
                giip = rs0*stoiip 
            if giip>0 and fgiip==0:
                fgiip = giip - sgiip
                # if print_log:
                #     print(f'gas cap volume: {fgiip:.3e} sm3')   
            if giip==0 and fgiip!=0:
                giip = fgiip + sgiip

            if fgiip/giip < -1e-6:
                raise ValueError(
                    'Negative gas cap volume! \nCheck rs0, stoiip, fgiip etc.\n'+\
                    f'giip: {giip:.3e}\n'+ \
                    f'fgiip: {fgiip:.3e}\n'+ \
                    f'stoiip: {stoiip:.3e}\n'+ \
                    f'stoiip: {stoiip:.3e}\n'+ \
                    f'rs0: {rs0:.fe}\n'+ \
                    f'giip-stoiip*rs0=={giip-rs0*stoiip}')                

            oip[0] = stoiip
            rs[0] = rs0
            pb[0] = pb_of_rs(rs0)      

            gips[0] = sgiip
            gipf[0] = fgiip 

            bo[0] = fvf_oil(pb[0])*np.exp(co_of_pb(pb[0])*(pb[0]-p[0]))
            pv[0] += oip[0]*bo[0]
        
        else:
            # just in case
            if fgiip==0: fgiip=giip
            if giip==0:  giip=fgiip

        gip[0] = giip
        gipf[0] = fgiip
        pv[0] += gipf[0]*bg[0]

    # aquifer(s)
    we = np.zeros((num_steps,))  # water influx from aquifer
    we_cum = np.zeros((num_steps,))  # cumulative --//--//--
    if case.aquifer is not None:
        pi_aq = case.aquifer.pi
        if case.aquifer.model == 'Fetkovich':
            p_aq = np.zeros((num_steps,)) # preallocating
            if case.aquifer.p0 is None:
                p_aq[0] = case.reservoir.p0
            else:
                p_aq[0] = case.aquifer.p0

            if case.aquifer.ct is None:
                ct = cw + cf
            else:
                ct = case.aquifer.ct

            Ct_Vw0 = ct*case.aquifer.v
            pi_Ct_Vw0 = pi_aq/Ct_Vw0
            wei = Ct_Vw0*p_aq[0]
            
    frv = pv.copy()
    err_pr = np.zeros((num_steps,))
    err_rv = np.zeros((num_steps,))
    err_we = np.zeros((num_steps,))
    err_co2i = np.zeros((num_steps,))
    h = 1e-8
# ------------------------------------------------------------------------
    def compute_mbe_discrepancy(p_):
        f = 0  # discrepancy
        A, B, C, D, E = 0, 0, 0, 0, 0
        # A = PV
        # B - fluid volumes in RC at the previous step
        # C - fluid expansion
        # D - produced volumes (RC)
        # E - dissolution/resolution
        pb_, rs_, bo_, bw_, bg_, bco2_ = 0, 0, 0, 0, 0, 0
        #A += pv[i-1]*(1-cf*(p[i-1] - p_))
        # A += pv[0]*(1-cf*(p[0] - p_))
        A += pv[0]*pv_mult_func(p_-p[0])

        if gas:
            bg_ = fvf_gas(p_)
            B += bg[i-1]*gipf[i-1]
            C += gipf[i-1]*(bg_ - bg[i-1])
            D -= ng[i]*bg_

            if oil:
                rss_ = rs_of_pb(p_)
                if p_ > pb[i-1] and stick_to_rs0==False:  # gas dissolution
                    # dissolution rate limit
                    rs_lim1 = rs[i-1] + case.fluids.oil.drsdt*dt 
                    # dissolution tapering coefficient
                    rs_lim2 = rs[i-1] + case.fluids.oil.dtc*(rss_ - rs[i-1])
                    # to prevent Rs from increasing beyond initially available gas cap volume
                    rty = (gip[i]-gipf[0]*(1-case.fluids.oil.dvf))/oip[i]
                    rs_lim3 = max([rty, rs[i-1]])
                    del rty
                    rs_ = min([rs_lim1, rs_lim2, rs_lim3])
                else:  # gas comes out solution
                    rs_ = min([rs[i-1], rss_])

                pb_ = pb_of_rs(rs_)
                # bo_ = fvf_oil(pb_)*(1 - (p_-pb_)*co_of_pb(pb_)) #!!
                bo_ = fvf_oil(pb_)*np.exp((pb_-p_)*co_of_pb(pb_))

                B += oip[i-1]*bo[i-1]
                C += oip[i-1]*(bo_ - bo[i-1])
                D -= no[i]*bo_ 
                # evolved/dissolved gas
                E = bg_*(oip[i-1]*rs[i-1] - oip[i]*rs_)

        if water:
            bw_ = fvf_wat(p_)
            B += wip[i-1]*bw[i-1]
            C += wip[i-1]*(bw_ - bw[i-1])
            D -= nw[i]*bw_

        if co2:
            bco2_ = fvf_co2(p_)
            B += co2ip[i-1]*bco2[i-1]
            C += co2ip[i-1]*(bco2_ - bco2[i-1])
            D -= nco2[i]*bco2_

        # aquifer influx
        we_ = we0
        if (case.aquifer is not None):
            # wei = (cw+cf)*p_aq[0]*case.aquifer.v
            if (case.aquifer.model == 'Fetkovich'):
                if case.aquifer.mode == 'modified':
                    we_ = (p_aq[i-1] - p_)*Ct_Vw0*(1 - np.exp(-pi_Ct_Vw0*dt))
                elif case.aquifer.mode == 'original':
                    p_av_ = 0.5*(p[i-1] + p_)  # average reservoir pressure
                    we_ = (p_aq[i-1] - p_av_)*Ct_Vw0*(1 - np.exp(-pi_Ct_Vw0*dt))                       
                else:
                    raise ValueError(f'{case.aquifer.mode} is unknown formulation for the Fetkovich aquifer!')
            elif (case.aquifer.model == 'SteadyState'):
                we_ = (p[0] - 0.5*(p[i-1]+p_))*pi_aq*dt
            else:
                raise ValueError(f'{case.aquifer.model} is unknown aquifer model!')

        f = A - B - C - D - E - we_*bw_

        return f, A, D, E, rs_, pb_, bo_, bw_, bg_, bco2_, we_
# ------------------------------------------------------------------------
    for i in case.flows.totals.index[1:]:

        dt = t[i] - t[i-1]            
        # initial guess
        we0 = we[i-1]
        p_ = p[i-1]

        if gas:
            gip[i] = gip[i-1]-ng[i]
            if gip[i] < 0:
                if print_warnings:
                    warn('\n   gas in-place volume is negative!!! Run is stopped!')
                break  
            if oil:
                oip[i] = oip[i-1]-no[i]
                if oip[i] < 0:
                    if print_warnings:
                        warn('\n   oil in-place volume is negative!!! Run is stopped!')
                    break              
                # # alternative initial guess for undersaturated oil cases:
                # if gipf[i-1]/gip[0] < 1e-6:
                #     p_ = min(pb_of_rs(gip[i]/oip[i]),p_)

        pj0 = p_    
        f, A, D, E, rs_, pb_, bo_, bw_, bg_, bco2_, we_ = \
            compute_mbe_discrepancy(p_)
        ee = abs(f/A)

        j = 1
        while ee > max_error and j <= jmax:
            fh, A, D, E, rs_, pb_, bo_, bw_, bg_, bco2_, we_ = \
                 compute_mbe_discrepancy(p_ + h)
            drv = (fh - f)/h
            p_ = p_ - f/drv

            if p_ < 0:  raise ValueError(f'p={p_}<0 @ iter {i}!')

            # finalizing the current iteration // moving to the next one  ...
            f, A, D, E, rs_, pb_, bo_, bw_, bg_, bco2_, we_ = \
                compute_mbe_discrepancy(p_)
            # calculation of errors:
            if we_ != 0:
                err_we[i] = abs((we_ - we0)/we_)
            err_pr[i] = abs(p_ - pj0)  # pressure error
            err_rv[i] = abs(f/A)
            ee = err_rv[i]
            # ee = err_pr[i]
            if print_log:
                print(
                    f'   step(iter) {i:04}({j:1}) | {t[i]:.1f} days, p={p_:.3e} bar | error={ee:.2e} ')
            pj0 = p_  # !
            we0 = we_

            if oil and rs_< rs[0]: stick_to_rs0 = False

            j += 1

        p[i] = p_
        pv[i] = A
        nwrrc[i] = -D/dt
        del A, D, E
        frv_ = 0
        if gas:
            bg[i] = bg_
            gipf[i] = gip[i]
            if oil:
                rs[i] = rs_
                pb[i] = pb_
                bo[i] = bo_
                gips[i] = oip[i]*rs_
                gipf[i] = gip[i] - gips[i]
                # if rs_ != rs[i-1]:
                #     gips[i] = oip[i]*rs_
                #     gipf[i] = gip[i] - gips[i]
                # else:
                #     gipf[i] == 0                
                frv_ += oip[i]*bo_          
            frv_ += gipf[i]*bg[i]
            if gipf[i]/gip[0] < -1e-6:   
                if print_warnings:
                    warn('\n   free gas in-place volume is negative!!! Run is stopped!'+\
                         f'\n   gipf[i]/gip[0]: {gipf[i]/gip[0]}, gipf[i] : {gipf[i]}')            
                break

        if water:
            bw[i] = bw_
            we[i] = we_
            we_cum[i] = we_cum[i-1] + we[i]
            wip[i] = wip[i-1] - nw[i]
            wip[i] = wip[i] + we[i]
            if wip[i] < 0:
                if print_warnings:
                    warn('\n   water in-place volume is negative!!! Run is stopped!')
                break      

            frv_ += wip[i]*bw_

        if co2:
            bco2[i] = bco2_
            co2ip[i] = co2ip[i-1] - nco2[i]
            frv_ += co2ip[i]*bco2_
            if co2ip[i] < 0:
                if print_warnings:
                    warn('\n   CO2 in-place is negative!!! Run is stopped!')
                break                

        if case.aquifer is not None:
            if case.aquifer.model == 'Fetkovich':
                p_aq[i] = p_aq[0]*(1 - we_cum[i]/wei)
        
        frv[i] = frv_

        if p_ < 1.01325:
            if print_warnings:
                warn(f'\n   The run is stopped @{t[i]} as days reservoir pressure dropped below 1 atm ({p_} bar!')
            break

    # saving results into DataFrame
    results_dict = {}
    results_dict['days'] = t
    results_dict['years'] = t/365.25
    results_dict['date'] = case.flows.totals['date']
    results_dict['pressure (bar)'] = p
    results_dict['PV (rm3)'] = pv

    if case.aquifer is not None:
        results_dict['cumulative aquifer influx (sm3)'] = we_cum
        results_dict['aquifer influx rate (sm3/day)'] = np.zeros((num_steps,))
        results_dict['aquifer influx rate (sm3/day)'][1:] = np.diff(
            we_cum)/np.diff(t)
        results_dict['aquifer influx rate (rm3/day)'] =\
            results_dict['aquifer influx rate (sm3/day)']*bw
        if case.aquifer.model == 'Fetkovich':
            results_dict['aquifer pressure (bar)'] = p_aq

    if gas:
        results_dict['GIP (sm3)'] = gip
        results_dict['free GIP (sm3)'] = gipf
        results_dict['dissolved GIP (sm3)'] = gips
        results_dict['gas FVF (rm3/sm3)'] = bg
        if oil:
            results_dict['OIP (sm3)'] = oip
            results_dict['Rs (sm3/sm3)'] = rs
            results_dict['p_bub (bar)'] = pb
            results_dict['oil FVF (rm3/sm3)'] = bo

    if co2:
        results_dict['CO2IP (sm3)'] = co2ip
        results_dict['CO2 FVF (rm3/sm3)'] = bco2

    if water:
        results_dict['WIP (sm3)'] = wip
        results_dict['water FVF (rm3/sm3)'] = bw

    if request_qQ:
        # adding rates (RC), rates (SC) and totals in the results
        cols = case.flows.rates.columns
        for mm,mm2 in zip(['prod','inj'], ['production','injection']):    
            for fl,fl2 in zip(['oil','wat','gas','co2'], ['oil','water','gas','CO2']):
                c_q = f'{fl}_{mm}'
                c_qrc = f'{fl2} {mm2} rate in RC (rm3/day)'
                c_qsc = f'{fl2} {mm2} rate in SC (sm3/day)'
                c_Qsc = f'{fl2} {mm2} total (sm3)'
                c_fvf = f'{fl2} FVF (rm3/sm3)'
                if c_q in cols:
                    results_dict[c_qrc] = results_dict[c_fvf]*case.flows.rates[c_q]
                    results_dict[c_qsc] = case.flows.rates[c_q]
                    results_dict[c_Qsc] = case.flows.totals[c_q]

    results_dict['net withdrawal rate (rm3/day)'] = nwrrc
    results_dict['err_rv'] = err_rv
    case.results = pd.DataFrame(data=results_dict)

    # in case of crash:
    if i !=  case.flows.totals.index[-1]:
        if print_warnings: warn('the results are trimmed as a crash happened')
        case.results = case.results.loc[:i-1,:] 

    return case


def usc_run(case, print_log=False, print_warnings=True):
    '''evaluate of Ultimate Storage Capacity (USC)
    
    Parameters:
    ----------
    self : unicellar.Case
        material balance model with USC scenarios in case.usc
        use usc_template function to generate a prefilled template
    print_log : bool, optional
        print log or not
    print_warnings : bool, optional
        print warnings or not        

    Returns
    --------
    self.usc : pd.DataFrame
        columns with storage capacities are added: 'USC (rm3)','USC (sm3)','USC (t)'
    '''
    cf = case.reservoir.cf  # !!!
    oil = (True if case.fluids.oil is not None else False)
    gas = (True if case.fluids.gas is not None else False)
    water = (True if case.fluids.water is not None else False)
    co2 = (True if case.fluids.co2 is not None else False)

    p0 = case.reservoir.p0
    pv0 = 0  # pore volume
    if water:
        cw = case.fluids.water.cw
        bw_ref = case.fluids.water.fvf_ref
        pw_ref = case.fluids.water.p_ref        
        fvf_wat = lambda p: bw_ref*np.exp(cw*(pw_ref - p))

        wip0 = case.reservoir.wiip
        bw0 = fvf_wat(p0)
        pv0 += wip0*bw0

    if co2:
        x = case.fluids.co2.fvf_table.iloc[:, 0]
        y = case.fluids.co2.fvf_table.iloc[:, 1]
        rep_fvf_co2 = interp1d(x, 1/y, fill_value="extrapolate")
        def fvf_co2(x): return 1/rep_fvf_co2(x)
        co2_den_sc = case.fluids.co2.density
        del x, y
        co2ip0 = case.reservoir.co2iip
        bco20 = fvf_co2(p0)
        pv0 += co2ip0*bco20

    if gas:
        x = case.fluids.gas.fvf_table.iloc[:, 0]
        y = case.fluids.gas.fvf_table.iloc[:, 1]
        rep_fvf_gas = interp1d(x, 1/y, fill_value="extrapolate")
        def fvf_gas(x): return 1/rep_fvf_gas(x)
        del x, y

        gip0 = case.reservoir.giip
        gipf0 = case.reservoir.giip  # is corrected below
        bg0 = fvf_gas(p0)
        # oil should always contain gas
        if oil:
            if case.reservoir.rs0 is None:
                rs0 = rs_of_pb(p[0])
                if print_log:
                    print( f'rs0 is not specified => estimated from rs_table: {rs0:.2f} sm3/sm3')
            else:
                rs0 = case.reservoir.rs0            
            # FVF vs. p_bub
            x = case.fluids.oil.fvf_table.iloc[:, 0]  # pb
            y = case.fluids.oil.fvf_table.iloc[:, 1]  # FVF
            fvf_oil = interp1d(x, y, kind='linear', fill_value="extrapolate")
            # oil compressibility vs. p_bub
            x = case.fluids.oil.co_table.iloc[:, 0]
            y = case.fluids.oil.co_table.iloc[:, 1]
            co_of_pb = interp1d(x, y, fill_value="extrapolate")
            del x, y          
            pb = case.fluids.oil.rs_table.iloc[:, 0]
            rs = case.fluids.oil.rs_table.iloc[:, 1]
            pb_of_rs = interp1d(rs, pb, kind='linear',
                                fill_value="extrapolate")
            rs_of_pb = interp1d(pb, rs, kind='linear',
                                fill_value="extrapolate")
            del rs, pb
            oip0 = case.reservoir.stoiip
            pb0 = pb_of_rs(rs0)
            bo0 = fvf_oil(pb0)*(1 - (p0 - pb0)*co_of_pb(pb0))
            gips0 = oip0*rs0
            gipf0 -= gips0
            if gipf0 < 0:
                raise ValueError(
                    'gipf0<0 free gas is negative /n check the input rs')
            pv0 += oip0*bo0
        pv0 += gipf0*bg0

    # aquifer(s)
    we = 0  # water influx from aquifer
    if case.aquifer is not None:
        if (case.aquifer.model == 'Fetkovich'):
            
            if case.aquifer.p0 is None:
                p_aq0 = case.reservoir.p0
            else:
                p_aq0 = case.aquifer.p0

            if case.aquifer.ct is None:
                ct = cw + cf
            else:
                ct = case.aquifer.ct  

            wei = ct*p_aq0*case.aquifer.v          

    # compaction model
    compaction_model = case.reservoir.compaction_model
    if isinstance(compaction_model, str):
        compaction_model = compaction_model.lower()
        if compaction_model == 'cf':
            pv_mult_func = lambda dp: np.exp(cf*dp)
        elif compaction_model == 'pv table' or compaction_model == 'pv_table':
            pv_table = case.reservoir.pv_table.copy()
            if isinstance(case.reservoir.pv_table, pd.DataFrame):
                pv_table = pv_table.to_numpy()
            dp = pv_table[:, 0]
            pv_mult = pv_table[:, 1]
            pv_mult_func = interp1d(dp, pv_mult, kind='linear')
        else:
            raise ValueError(
                f'compaction model "{compaction_model}" is unknown!')
    elif callable(compaction_model):
        pv_mult_func = compaction_model
    else:
        raise ValueError(f'compaction model is in a wrong format!')

    for i in case.usc.index:

        p = case.usc.p_max[i]
        op, wp, gp, co2p = 0, 0, 0, 0
        oi, wi, gi, co2i = 0, 0, 0, 0
        # preallocating oil, water, gas, CO2 production and injection
        if 'oil_prod' in case.usc.columns:
            op = case.usc['oil_prod'].loc[i]
        if 'wat_prod' in case.usc.columns:
            wp = case.usc['wat_prod'].loc[i]
        if 'gas_prod' in case.usc.columns:
            gp = case.usc['gas_prod'].loc[i]
        if 'co2_prod' in case.usc.columns:
            co2p = case.usc['co2_prod'].loc[i]
        if 'oil_inj' in case.usc.columns:
            oi = case.usc['oil_inj'].loc[i]
        if 'wat_inj' in case.usc.columns:
            wi = case.usc['wat_inj'].loc[i]
        if 'gas_inj' in case.usc.columns:
            gi = case.usc['gas_inj'].loc[i]
        if 'co2_inj' in case.usc.columns:
            co2i = case.usc['co2_inj'].loc[i]

        # net withdrawal
        no = op - oi  # oil
        nw = wp - wi  # water
        ng = gp - gi  # gas
        nco2 = co2p - co2i  # CO2

        if oil:
            oip = oip0 - no
        if gas:
            gip = gip0 - ng

        A, B, C, D, E = 0, 0, 0, 0, 0
        pb, rs, bo, bw, bg, bco2 = 0, 0, 0, 0, 0, 0
        A += pv0*pv_mult_func(p-p0)
        
        if gas:
            bg = fvf_gas(p)
            B += bg0*gipf0
            C += (bg - bg0)*gipf0
            D -= ng*bg

            if oil:
                rss = rs_of_pb(p)
                if p > pb0:  # gas dissolution
                    # to prevent Rs from increasing beyond initially available gas cap volume
                    rty = (gip-gipf0*(1-case.fluids.oil.dvf))/oip
                    rs_lim = max([rty, rs0])
                    del rty
                    rs = min([rss, rs_lim])
                else:  # gas comes out solution
                    rs = min([rs0, rss])

                pb = pb_of_rs(rs)
                bo = fvf_oil(pb)*(1 - (p - pb)*co_of_pb(pb))

                B += oip0*bo0
                C += oip0*(bo - bo0)
                D -= no*bo  # D -= no*(bo - rs*bg)
                E = bg*(oip0*rs0 - oip*rs)  # evolved/dissolved gas

        if water:
            bw = fvf_wat(p)
            B += wip0*bw0
            C += wip0*(bw - bw0)
            D -= nw*bw

        if co2:
            bco2 = fvf_co2(p)
            B += co2ip0*bco20
            C += co2ip0*(bco2 - bco20)
            D -= nco2*bco2

        # aquifer influx
        we = 0
        if (case.aquifer is not None):
            if (case.aquifer.model == 'Fetkovich'):
                we = wei*(1 - p/p_aq0)

            if (case.aquifer.model == 'SteadyState'):
                if print_warnings:
                    warn('\n   Steady State aquifer is not fully supported. zero influx assummed!')

        dpv = A - B - C - D - E - we*bw
        if oil: 
            case.usc.loc[i, 'Rs'] = rs
        case.usc.loc[i, 'USC (rm3)'] = dpv
        case.usc.loc[i, 'USC (sm3)'] = dpv/bco2
        case.usc.loc[i, 'USC (t)'] = dpv/bco2*co2_den_sc/1000

    return case


def usc_template(flows=None, reservoir=None):
    '''create a template for scenarios to estimate Ultimate Storage Capacity 

    Parameters
    ----------    
    (may be used to prefill template)
    flows : Flows class instance which describes production/injection history
        see Flows class description
    reservoir : Reservoir class instance which describes reservoir properties
        see Flows class description

    Returns
    -------
    pd.DataFrame with storage scenarios in rows and the following columns:
    * p_max - maximal pressure (bar)
    * oil_prod - oil production (sm3)
    * wat_prod - water  production (sm3)
    * gas_prod - gas net production (sm3)
    * co2_prod - CO2 net production (sm3)
    * oil_inj - oil injection (sm3)
    * wat_inj - water injection (sm3)
    * gas_inj - gas injection (sm3)
    * co2_inj - CO2 injection (sm3)
    * fluid - storage fluid (at the moment reserved for CO2 only)

    By default two scenarios with zero and None values are created.
    If Flows or/and Reservoir instances are provided the template is prefilled.
    If flows!=None => produced/injected total volumes for the second scenario 
    are retrieved from the last entry
    IF reservoir!=None => maximal pressure is set to:
    * reservoir.p0 for the first scenario
    * reservoir.p0+25% for the second scenario
    '''

    df = pd.DataFrame({'p_max': [None, None],
                       'oil_prod': [0, 0],
                       'gas_prod': [0, 0],
                       'wat_prod': [0, 0],
                       'co2_prod': [0, 0],
                       'oil_inj':  [0, 0],
                       'gas_inj':  [0, 0],
                       'wat_inj':  [0, 0],
                       'co2_inj':  [0, 0],
                       'comment': ['scenario #0', 'scenario #1']
                       })
    df['fluid'] = 'CO2'

    if reservoir is not None:
        p0 = reservoir.p0
        df.loc[0, 'p_max'] = p0
        df.loc[1, 'p_max'] = p0*1.25

    # copying the cumulatives ...
    if flows is not None:
        for i in flows.totals.columns:
            if ('prod' in i) | ('inj' in i):
                df.loc[1, i] = flows.totals[i].iloc[-1]

    return df
