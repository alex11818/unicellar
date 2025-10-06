# spell check
# cSpell:enable
# cSpell:includeRegExp #.*
# cSpell:includeRegExp /(["]{3}|[']{3})[^\1]*?\1/g

"""
collection of preprocessing functions: 1) to facilitate model setup and
2) to make various guesstimates
"""

import pickle
import pandas as pd 
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os, requests, io
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

# %% Auxiliary functions ------------------------------------------------------
def read_pickle(file_path):
    '''load content from a pickle (*.pkl) file 

    Parameters
    ----------
    file_path : str
        path to a file

    Returns
    -------
    unpickled : same type as object stored in file
    '''
    if file_path[-4:] != '.pkl':
        file_path = file_path + '.pkl'
    with open(file_path, "rb") as fff:
        foo = pickle.load(fff)
    return foo

def estimate_bhp(fluid='CO2', thp=54.996991, tvd=1000,
                 degC=20,
                 all_points=False,
                 max_abs_error=1e-7,
                 num_points=None,
                 max_iter=10):
    '''
    estimate BHP as THP + column weight (i.e. ignoring friction)

    Parameters
    ----------
    thp : float, int
        Tubing Head Pressure (bar)
    degC : float, int
        representative temperature (degC)
    tvd :  float, int
        well True Vertical Depth (m)
    all_points: bool
        if True a pd.DataFrame with data for all intermediate points is returned
    max_abs_error: float
        maximum absolute pressure error (bar)
    num_points: int
        number of discretization points (>1)
        Default: round(tvd) + 1

    Returns
    ----------
    if all_points==True:
        pd.DataFrame with depth, density and pressure for all intermediate points
    else:
        BHP: float
    '''

    table = get_fluid_properties(fluid, degC)
    den_func = interp1d(table['Pressure (bar)'], table['Density (kg/m3)'])

    if num_points is None:
        num_points = round(tvd) + 1

    z_arr = np.linspace(0, tvd, num_points)
    dz = z_arr[1] - z_arr[0]

    p = thp
    p_arr = [p]
    rho = den_func(p)
    rho_arr = [rho]
    cf = 9.81/1e+5

    for z in z_arr[1:]:
        # fixed point iteration
        dp = 0  # forcing to make an iteration
        dp_ = dz*rho*cf
        ic = 1
        while abs(dp - dp_) > max_abs_error and ic <= max_iter:
            rho_ = den_func(p + dp_)
            dp = dp_
            dp_ = 0.5*cf*dz*(rho + rho_)
            ic = ic + 1

        p = p + dp_
        rho = den_func(p)
        p_arr.append(p)
        rho_arr.append(rho)

    if all_points:
        return pd.DataFrame(
            data={'depth (m)': z_arr,
                  'density (kg/m3)': rho_arr,
                  'pressure (bar)': p_arr})
    else:
        return p


def estimate_thp(fluid='CO2', bhp=130, tvd=1000,
                 degC=20,
                 all_points=False,
                 max_abs_error=1e-7,
                 num_points=None,
                 max_iter=10):
    '''
    estimate THP as BHP - column weight (i.e. ignoring friction)

    Parameters
    ----------
    bhp : float, int
        Bottom Hole Pressure (bar)
    degC : float, int
        representative temperature (degC)
    tvd :  float, int
        well True Vertical Depth (m)
    all_points : bool
        if True a pd.DataFrame with data for all intermediate points is returned
    max_abs_error: float
        maximum absolute pressure error (bar)
    num_points: int
        number of discretization points (>1)
        Default: round(tvd) + 1

    Returns
    ----------
    if all_points==True:
        pd.DataFrame with depth, density and pressure for all intermediate points
    else:
        THP: float
    '''

    table = get_fluid_properties(fluid, degC = degC)
    den_func = interp1d(table['Pressure (bar)'], table['Density (kg/m3)'])

    if num_points is None:
        num_points = round(tvd) + 1

    z_arr = np.linspace(tvd, 0, num_points)
    dz = z_arr[1] - z_arr[0]

    p = bhp
    p_arr = [p]
    rho = den_func(p)
    rho_arr = [rho]
    cf = 9.81/1e+5

    for z in z_arr[1:]:
        # fixed point iteration
        dp = 0  # forcing to make an iteration
        dp_ = dz*rho*cf
        ic = 1
        while abs(dp - dp_) > max_abs_error and ic <= max_iter:
            rho_ = den_func(p + dp_)
            dp = dp_
            dp_ = 0.5*cf*dz*(rho + rho_)
            ic = ic + 1

        p = p + dp_
        rho = den_func(p)
        p_arr.append(p)
        rho_arr.append(rho)

    if all_points:
        return pd.DataFrame(
            data={'depth (m)': z_arr,
                  'density (kg/m3)': rho_arr,
                  'pressure (bar)': p_arr})
    else:
        return p


def estimate_compression_work(p0=1, p=200, degC=20, fluid='CO2', reload=True):
    '''
    estimate minimal (specific) work required by isothermal compressor
    (to compress and pump 1kg of FLUID from P0 to P bar pressure at temperature DEGC)

    Parameters
    ----------    
    p0 : float, int
        input pressure (bar)
    p : float, int, list, np.array
        output pressure(s) (bar)
    degC : float, int
        temperature (degC)
    fluid : str
        following fluids are available: 
        ('H2O', 'N2', 'H2', 'O2', 'CO2', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'C10H22')
    reload : bool
        (re)use/save standardized NIST table (if possible) to avoid 
        excessive requests to NIST webpage

    Returns
    -------
    w : float, ndarray 
        specific work for isothermal compressor (J/kg)
    q : float, ndarray
        specific waste heat (J/kg)
    '''
    if isinstance(p, float) or isinstance(p, int):
        p = np.array([p])

    p_max = max(p)
    if p_max <= 1011 and p0 == 1 and reload:
        table = get_fluid_properties(
            fluid, p_min=1, p_max=1011, p_inc=10, degC=degC)
    else:
        table = get_fluid_properties(
            fluid, p_min=p0, p_max=p_max, p_inc=10, degC=degC)

    h_func = interp1d(table['Pressure (bar)'], 1000*table['Enthalpy (kJ/kg)'])
    s_func = interp1d(table['Pressure (bar)'], 1000*table['Entropy (J/g*K)'])

    s1 = s_func(p0)
    s2 = s_func(p)

    h1 = h_func(p0)
    h2 = h_func(p)

    q = -(s2 - s1)*(273.15 + degC)  # specific waste heat
    w = h2 - h1 + q  # specific compression work

    if len(q) == 1:
        w, q = w[0], q[0]

    return w, q


def get_fluid_properties(fluid='CO2', p_min=1, p_max=1011, p_inc=10, degC=20,
                         den_sc=None, process=True, nist_folder='nist_tables'):
    '''to fetch fluid property table from NIST Chemistry WebBook [1] and save it
    (by default) in a folder (created if needed inside the current directory)
    
    Reference:
    1. Eric W. Lemmon, Ian H. Bell, Marcia L. Huber, and Mark O. McLinden, 
    "Thermophysical Properties of Fluid Systems"  in NIST Chemistry WebBook, 
    NIST Standard Reference Database Number 69, 
    Eds. P.J. Linstrom and W.G. Mallard, National Institute of Standards and Technology, 
    Gaithersburg MD, 20899, https://doi.org/10.18434/T4D303, (retrieved November 29, 2022).

    Parameters
    ----------
    fluid : str
        name or chemical formula:
        'H2O', 'N2', 'H2', 'O2', 'CO2', 'CH4', 'C2H6', 'C3H8', 'C4H10', 'C10H22'
            or
        'water', 'nitrogen', 'hydrogen', 'oxygen', 'carbon dioxide', 'methane',
        'ethane', 'propane', 'butane', 'decane'
    p_min, p_inc, p_max : 
        minimum pressure, pressure increment, maximum pressure (bar)
    degC :
        temperature (degC)
    den_sc : float, int
        fluid density (kg/m3) in SC (standard conditions) to calculate FVF
        (formation volume factor)
        SC definitions:
        - ISO 13443: 1 atm and 15 degC, den_sc(CO2) = 1.8718 kg/m3
        - SPE: 1 bar and 15 degC, den_sc(CO2) = 1.8472 kg/m3
    process : boolean
        if True => duplicates around critical point are removed
    nist_folder : str
        folder to save (and then, if needed, to upload NIST tables)
        nist_folder==None => the option is switched off

    Returns
    -------
    NIST table formatted as pd.DataFrame 
    '''
    dict1 = {'H2O': 'C7732185',
             'N2': 'C7727379',
             'H2': 'C1333740',
             'O2': 'C7782447',
             'CO2': 'C124389',
             'CH4': 'C74828',
             'C2H6': 'C74840',
             'C3H8': 'C74986',
             'C4H10': 'C106978',
             'C10H22': 'C124185'}

    dict2 = {'water': 'C7732185',
             'nitrogen': 'C7727379',
             'hydrogen': 'C1333740',
             'oxygen': 'C7782447',
             'carbon dioxide': 'C124389',
             'methane': 'C74828',
             'ethane': 'C74840',
             'propane': 'C74986',
             'butane': 'C106978',
             'decane': 'C124185'}

    TT = pd.DataFrame()
    nist_id = None
    if fluid in dict1.keys():
        nist_id = dict1[fluid]
    if fluid in dict2.keys():
        nist_id = dict2[fluid]

    if nist_id == None:
        print('Fluid {0} was not found in neither dictionary'.format(fluid))
        print('please use:')
        print(dict1.keys())
        print('    or    :')
        print('please use:')
        print(dict2.keys())
    else:
        make_request = True
        if nist_folder is not None:
            out_file = nist_folder + \
                f'\\{fluid}_p_min={p_min}_p_max={p_max}_p_inc={p_inc}_degC={degC}.csv'
            if not os.path.isdir(nist_folder):
                os.mkdir(nist_folder)
                print(fr'"{nist_folder}" is created')
            else:
                if os.path.isfile(out_file):
                    TT = pd.read_csv(out_file)
                    make_request = False

        if make_request:
            # New URL format
            rs = ('https://webbook.nist.gov/cgi/fluid.cgi?'
                  + 'T={0}&'
                  + 'PLow={1}&'
                  + 'PHigh={2}&'
                  + 'PInc={3}&'
                  + 'Digits=5&'
                  + 'ID={4}&'
                  + 'Action=Data&'  # Use 'Data' not 'Load' to get CSV
                  + 'Type=IsoTherm&'
                  + 'TUnit=C&'
                  + 'PUnit=bar&'
                  + 'DUnit=kg%2Fm3&'
                  + 'HUnit=kJ%2Fkg&'
                  + 'WUnit=m%2Fs&'
                  + 'VisUnit=uPa*s&'
                  + 'STUnit=N%2Fm&'
                  + 'RefState=DEF')
            rs = rs.format(degC, p_min, p_max, p_inc, nist_id)
            # print(rs) 
            response = requests.get(rs)
            response.raise_for_status()  # Raise an error for bad status codes
            TT = pd.read_csv(io.StringIO(response.text), delimiter='\t')
            if nist_folder is not None:
                TT.to_csv(out_file, index=False)

        if process:
            # dropping duplicated pressures near the critical point
            TT.drop_duplicates(
                subset=['Pressure (bar)'], keep='last', inplace=True)

        # adding formation volume factor (RC/SC)
        if den_sc is not None:
            TT['FVF'] = den_sc/TT['Density (kg/m3)']

    return TT


# %% Eclipse related functions ------------------------------------------------
def read_rsm(somepath, reload=False, description=None, print_log=True):
    '''reads and pickles eclipse results from RSM file

    Parameters
    ----------
    somepath : str
        path to a RSM or DATA file
    reload : bool, optional
        reload or not available results from pickle file
    description : str, dict, list etc., optional
        description to be placed into INFO (see below)
    print_log : bool, optional
        print log or not         

    Returns
    ----------
    DF : pd.DataFrame with results
    units : dict with units
    info : dict with meta data: file path, name and header

    '''
    p = somepath.replace('.DATA', '.RSM')
    p = p.replace('.data', '.RSM')
    p_out = p.replace('.RSM', '.pkl')
    if reload == True and os.path.exists(p_out):
        try:
            # let's try to read the pickle 
            with open(p_out, "rb") as F:
                DF, units, info = pickle.load(F)
            if print_log: print('eclipse results are loaded from ' + p_out)
            return DF, units, info                
        except:
            if print_log: 
                print(f'some error occured while reading {p_out}')
                print(f'trying to read the RSM file instead ...')

    f = open(p, 'r')
    DF = pd.DataFrame()
    units = dict()
    a = f.readline()
    c = 1
    if a[0] == '1':
        while True:
            a = f.readline()  # passing  -----------------------------------
            h = f.readline()  # parsing the headline
            a = f.readline()  # passing  -----------------------------------
            a = f.readline()  # passing  header: 'TIME      FOPR      FLPR ....'
            V = a.split()     # getting a list with vector names
            # getting the column width:
            aa = a.strip()
            for i in range(8, 15):
                if aa[i] != ' ':
                    cw = i
                    break

            U = f.readline()  # for parsing  units: ' HOURS     SCC/HOUR  SCC/HOUR  SCC...'
            ncol = len(V)
            # for units multiplicators: '  *10**6       *10**6       *10**6'
            U2 = ' '*len(U)
            W = f.readline()      # for parsing well names if applicable
            if '*' in W:  # check for units multiplicators
                U2 = W
                W = f.readline()   # for parsing well names if applicable
            
            # for parsing numeric components of the vector names if applicable
            N = f.readline()      
            mults = np.ones((ncol,))
            for i in range(ncol):
                w = W[(1+i*cw):(1+cw*(i+1))].strip().\
                    replace('\n','').replace('  ', ' ').replace('  ', ' ').replace(' ', '_')
                n = N[(1+i*cw):(1+cw*(i+1))].strip().\
                    replace('\n','').replace('  ', ' ').replace('  ', ' ').replace(' ', '_')
                u = U[(1+i*cw):(1+cw*(i+1))].strip()
                u2 = U2[(1+i*cw):(1+cw*(i+1))].strip()
                if w != '':
                    V[i] = V[i] + '_' + w
                if n != '':
                    V[i] = V[i] + '_' + n
                # units[V[i]] = u + u2  # updating unit dictionary
                units[V[i]] = u # updating unit dictionary
                if len(u2) > 0:
                    mults[i] = eval(u2[1:])

            a = f.readline()  # passing  -----------------------------------
            # reading numerical values:
            a = f.readline()
            df = pd.DataFrame()
            while True:
                #  x = list(map(float,a.split()))
                #  x = list(map(float,a.split()))
                # have to use the cycle below to handle strings/empty entries like :(
                # -------------------------------------------------------------------------------------------------------------------------------
                #  TIME         NLINEARS     STEPTYPE     TELAPLIN     MSUMNEWT     TIMESTRY     NLINSMIN     NLINSMAX
                #  DAYS                                   SECONDS                   DAYS

                #  -------------------------------------------------------------------------------------------------------------------------------
                #         0            0                         0            0            0            0            0
                #  1.000000     2.000000       INIT       0.391250            2     1.000000            2            2
                #  4.000000     1.500000       MAXF       0.224714            4     3.000000            1            2
                x = []
                for i in range(ncol):
                    aa = a[(1+i*cw):(1+cw*(i+1))].strip()
                    try:
                        # convert to a number format if possible
                        aa = eval(aa)*mults[i]
                    except:
                        pass  # leave a string
                    x.append(aa)

                #df = df.append(pd.DataFrame([x], columns=V))
                df = pd.concat([df, pd.DataFrame([x], columns=V)])
                a = f.readline()
                if a == '':
                    break
                if a[0] == '1':
                    break
            if 'TIME' in df.columns:
                df.set_index('TIME', inplace=True)
            else:
                #df.set_index('DATE', inplace=True)
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.set_index('DATE', inplace=True)

            DF = pd.concat([DF, df], axis=1)
            # if a == '' or a =='1\n': break
            if a == '': break
    else:
        # to handle output from EXCEL keyword
        #             print('RSM with EXCEL keyword')
        while True:
            h = f.readline().replace('\t',' ')  # parsing the headline
            a = f.readline().replace('\t',' ')  # passing  header: 'TIME      FOPR      FLPR ....'
            V = a.split()     # getting a list with vector names
            # getting the column width:
            aa = a.strip()
            for i in range(8, 15):
                if aa[i] != ' ':
                    cw = i
                    break
            U = f.readline().replace('\t',' ')  # for parsing  units: ' HOURS     SCC/HOUR  SCC/HOUR  SCC...'
            ncol = len(V)
            # for units multiplicators: '  *10**6       *10**6       *10**6'
            U2 = ' '*len(U)
            W = f.readline().replace('\t',' ')      # for parsing well names if applicable
            if '*' in W:  # check for units multiplicators
                U2 = W
                W = f.readline().replace('\t',' ')      # for parsing well names if applicable
            N = f.readline().replace('\t',' ')      # for parsing numeric components of the vector names if applicable
            mults = np.ones((ncol,))
            for i in range(ncol):
                w = W[(1+i*cw):(1+cw*(i+1))].strip().\
                    replace('\n','').replace('  ', ' ').replace('  ', ' ').replace(' ', '_')
                n = N[(1+i*cw):(1+cw*(i+1))].strip().replace('\n','').\
                    replace('  ', ' ').replace('  ', ' ').replace(' ', '_')
                u = U[(1+i*cw):(1+cw*(i+1))].strip()
                u2 = U2[(1+i*cw):(1+cw*(i+1))].strip()
                if w != '':
                    V[i] = V[i] + '_' + w
                if n != '':
                    V[i] = V[i] + '_' + n
                # units[V[i]] = u + u2  # updating unit dictionary
                units[V[i]] = u # updating unit dictionary
                if len(u2) > 0:
                    mults[i] = eval(u2[1:])                
            # reading numerical values:
            df = pd.DataFrame()
            a = f.readline().replace('\t',' ')
            while True:
                #  x = list(map(float,a.split()))
                # have to use the cycle below to handle strings/empty entries like :(
                # -------------------------------------------------------------------------------------------------------------------------------
                #  TIME         NLINEARS     STEPTYPE     TELAPLIN     MSUMNEWT     TIMESTRY     NLINSMIN     NLINSMAX
                #  DAYS                                   SECONDS                   DAYS

                #  -------------------------------------------------------------------------------------------------------------------------------
                #         0            0                         0            0            0            0            0
                #  1.000000     2.000000       INIT       0.391250            2     1.000000            2            2
                #  4.000000     1.500000       MAXF       0.224714            4     3.000000            1            2

                x = []
                for i in range(ncol):
                    aa = a[(1+i*cw):(1+cw*(i+1))].strip()
                    try:
                        # convert to a number format if possible
                        aa = eval(aa)*mults[i]
                    except:
                        pass  # leave a string
                    x.append(aa)

                #df = df.append(pd.DataFrame([x], columns=V))
                df = pd.concat([df, pd.DataFrame([x], columns=V)])
                a = f.readline().replace('\t',' ')
                if a == '':
                    break
                if a[0] == '\n':
                    break
#                 df.set_index('TIME', inplace=True)
            if 'TIME' in df.columns:
                df.set_index('TIME', inplace=True)
            else:
                df['DATE'] = pd.to_datetime(
                    df['DATE'])  # conversion to datetime
                df.set_index('DATE', inplace=True)
            DF = pd.concat([DF, df], axis=1)
            if a == '':  break
    f.close()

    # if ('TIME' in DF.columns) and (start_date is not None):
    #    DF['DATE'] = start_date + pd.to_timedelta(DF['TIME'], unit='day')

    # just in case: remove duplicated columns
    DF = DF.loc[:, ~DF.columns.duplicated()]

    info = {'header': h, 'name': h.split()[3],
        'file': p, 'description': description}

    with open(p_out, "wb") as F:
        pickle.dump((DF, units, info), F)

    if print_log: print('results are saved in ' + p_out)

    return DF, units, info


def read_pvdg(p, units='metric'):
    '''
    fetch gas fvf_table for material balance model from Eclipse PVDG keyword
    NB! Only the first table can be read

    Parameters
    ----------
    p : str
        path to a file with PVDG keyword
    units : str, 'metric'|'field',  optional
         field units are converted to metric

    Returns
    ----------
    fvf_table : pd.DataFrame
        Formation Volume Factor FVF (rm3/sm3) vs. pressure (bars)
    pvdg : pd.DataFrame with raw PVDG data 
    '''
    # reading the keyword
    f = open(p, 'r', errors='replace')

    pr, fvf, visc = [], [], []

    for line in f:
        a = line.split()
        if len(a) > 0 and a[0] == 'PVDG': break 

    if len(a) > 0 and a[0] == 'PVDG':
        
        c, c_max = 0, 10000
        while c < c_max:
            c += 1
            a = f.readline().split()
            # to pass empty strings and '--'
            while (len(a)==0) or (a[0][:2]=='--'):
                a = f.readline().split()

            # to remove '--abcdef sd' from lines like 'x y z --abcdef sd'
            for n,w in enumerate(a):
                if w[:2] == '--': 
                    a = a[:n]
                    break    
            
            # to stop
            if a[0] == '/': break

            pr.append(eval(a[0]))
            fvf.append(eval(a[1]))
            visc.append(eval(a[2]))
            if len(a) > 3 and a[3] == '/': break
        
        f.close()
    else:
        f.close()
        raise Exception(f'PVTO keyword was not found in {p}')

    # DataFrame with all the points
    pvdg = pd.DataFrame(data={'pressure': pr, 'fvf': fvf, 'visc': visc})
    pvdg0 = pvdg.copy()

    if units.lower() == 'field':
        cf_gor = 1.801175e-1  # from cf/bbl to m3/m3
        cf_psi2bar = 6.894757e-02
        pvdg.loc[:, 'fvf'] = pvdg.loc[:, 'fvf']/cf_gor/1000
        pvdg.loc[:, 'pressure'] = pvdg.loc[:, 'pressure']*cf_psi2bar

    pvdg.rename(columns={
                'pressure': 'pressure (bar)',
                'fvf': 'FVF (rm3/sm3)'},
                inplace=True)

    fvf_table = pvdg[['pressure (bar)', 'FVF (rm3/sm3)']]

    return fvf_table, pvdg0


def read_pvco(p, units='metric'):
    '''
    fetch gas fvf_table for material balance model from Eclipse PVCO keyword
    NB! Only the first table can be read

    Parameters
    ----------
    p : str
        path to a file with PVDG keyword
    units : str, 'metric'|'field',  optional
         field units are converted to metric

    Returns
    ----------
    fvf_table : pd.DataFrame
        FVF (2nd column, rm3/sm3) vs. bubble point pressure (1st column, bar)
    rs_table : pd.DataFrame
        GOR (2nd column, sm3/sm3) vs. bubble point pressure (1st column, bar)        
    co_table : pd.DataFrame
        compressibility (2nd column, 1/bar) vs. bubble point pressure (1st column, bar)
    pvco: pd.DataFrame 
        raw PVTO data  
    '''
    # reading the keyword
    f = open(p, 'r', errors='replace')

    pr, rs, fvf, visc, co, viscosibility = [], [], [], [], [], []

    for line in f:
        a = line.split()
        if len(a) > 0 and a[0] == 'PVCO': break 

    if len(a) > 0 and a[0] == 'PVCO':
        
        c, c_max = 0, 10000
        while c < c_max:
            c += 1
            a = f.readline().split()
            # to pass empty strings and '--'
            while (len(a)==0) or (a[0][:2]=='--'):
                a = f.readline().split()

            # to remove '--abcdef sd' from lines like 'x y z --abcdef sd'
            for n,w in enumerate(a):
                if w[:2] == '--': 
                    a = a[:n]
                    break    
            
            # to stop
            if a[0] == '/': break

            pr.append(eval(a[0]))
            rs.append(eval(a[1]))
            fvf.append(eval(a[2]))
            visc.append(eval(a[3]))
            co.append(eval(a[4]))
            viscosibility.append(a[5])

            if len(a) > 3 and a[3] == '/': break
        
        f.close()
    else:
        f.close()
        raise Exception(f'PVCO keyword was not found in {p}')

    # DataFrame with all the points
    pvco = pd.DataFrame(\
        data={'p': pr, 'rs': rs, 'fvf': fvf, 'visc': visc, 'co': co, \
              'viscosibility': viscosibility})

    if units.lower() == 'field':
        cf_gor = 1.801175e-1  # from cf/bbl to m3/m3
        cf_psi2bar = 6.894757e-02
        pvco.loc[:, 'fvf'] = pvco.loc[:, 'fvf']/cf_gor/1000
        pvco.loc[:, 'p'] = pvco.loc[:, 'p']*cf_psi2bar

    pvco.rename(columns={'p': 'pressure (bar)',
                         'rs': 'Rs (sm3/sm3)',
                         'fvf': 'FVF (rm3/sm3)',
                         'co': 'co (1/bar)'}, inplace=True)

    fvf_table = pvco[['pressure (bar)', 'FVF (rm3/sm3)']]
    rs_table = pvco[['pressure (bar)', 'Rs (sm3/sm3)']]
    co_table = pvco[['pressure (bar)', 'co (1/bar)']]    

    return fvf_table, rs_table, co_table, pvco


def read_pvto(p, units='metric', co_est_ind='0:', plot=True):
    '''
    fetch oil properties from Eclipse PVTO keyword
    NB! Only the first table can be read!
    NB! No "1*"-style entries! 
    NB! undersaturated oil compressibilities are linearly approximated and 
        not 100% accurate!

    Parameters
    ----------
    p : str
        path to a file with PVTO keyword
    units : str, 'metric'|'field'
        field units are converted to metric
    co_est : str, optional
        points from a PVTO branch to be used for estimation of undersaturated 
        oil compressibility, specified as their Python indices like "0:",
        "0:2" or ":-1"
        
    plot : bool
        plot or not oil properties (FVF, Rs, oil viscosity)

    Returns
    ----------
    fvf_table : pd.DataFrame
        FVF (2nd column, rm3/sm3) vs. bubble point pressure (1st column, bar)
    rs_table : pd.DataFrame
        GOR (2nd column, sm3/sm3) vs. bubble point pressure (1st column, bar)        
    co_table : pd.DataFrame
        compressibility (2nd column, 1/bar) vs. bubble point pressure (1st column, bar)
    pvto: pd.DataFrame
        raw PVTO data    
    '''
    
    ind1, ind2 = [], []
    rs, bo, pr, visc = [], [], [], []

    i1 = 0  # block index
    i2 = 0  # line index within a block

    f = open(p, 'r', errors='replace')
    for line in f:
        a = line.split()
        if len(a) > 0 and a[0] == 'PVTO': break 

    if len(a) > 0 and a[0] == 'PVTO':
        # reading blocks
        a = f.readline().split()
        # to pass empty strings and '--'
        while (len(a)==0) or (a[0][:2]=='--'):
            a = f.readline().split()

        c, c_max = 0, 10000
        
        while c < c_max:
            c += 1            
            # to remove '--abcdef sd' from lines like 'x y z --abcdef sd'
            for n,w in enumerate(a):
                if w[:2] == '--': 
                    a = a[:n]
                    break

            if a[0] == '/': break

            if a[-1] == '/':
                if len(a) == 5:
                    rs.append(eval(a[0]))
                    pr.append(eval(a[1]))
                    bo.append(eval(a[2]))
                    visc.append(eval(a[3]))
                else:
                    rs.append(rs[-1])
                    pr.append(eval(a[0]))
                    bo.append(eval(a[1]))
                    visc.append(eval(a[2]))
                ind1.append(i1)
                ind2.append(i2)
                i1 += 1
                i2 = 0
            else:
                if len(a) == 4:
                    rs.append(eval(a[0]))
                    pr.append(eval(a[1]))
                    bo.append(eval(a[2]))
                    visc.append(eval(a[3]))
                else:
                    rs.append(rs[-1])
                    pr.append(eval(a[0]))
                    bo.append(eval(a[1]))
                    visc.append(eval(a[2]))
                ind1.append(i1)
                ind2.append(i2)
                i2 += 1
            a = f.readline().split()
            # to pass empty strings and '--'
            while (len(a)==0) or (a[0][:2]=='--'):
                a = f.readline().split()

        f.close()
    else:
        f.close()
        raise Exception(f'PVTO keyword was not found in {p}')
        # return fvf_table, rs_table, co_table, pvto0

    # dataframe with all the points
    pvto = pd.DataFrame(data={'ind1': ind1, 'ind2': ind2,
                        'rs': rs, 'pressure': pr, 'fvf': bo, 'visc': visc})
    pvto0 = pvto.copy()

    if units.lower() == 'field':
        cf_bbl2m3 = 1.589873e-1
        cf_gor = 1.801175e-1  # from cf/bbl to m3/m3
        cf_psi2bar = 6.894757e-02
        pvto['pressure'] = pvto['pressure']*cf_psi2bar
        pvto['rs'] = pvto['rs']*cf_gor*1000

    # dataframe with bubble point pressure entries
    pvto2 = pvto.loc[pvto.ind2 == 0, :].copy()
    pvto2.set_index('ind1', inplace=True)
    pvto2.drop('ind2', axis=1, inplace=True)
    # Extracting compressibility:
    for i in reversed(pvto.ind1.unique()):
        temp = pvto.loc[pvto.ind1 == i, :].copy()
        temp = eval(f"temp.iloc[{co_est_ind},:]")
        if temp.shape[0] > 1:
            BB = temp.fvf.values
            pp = temp.pressure.values
            lse = linregress(pp-pp[0], -np.log(BB)+np.log(BB[0]))
            co=lse.slope   
        pvto2.loc[i, 'co'] = co

    if plot == True:

        co_of_pb = interp1d(pvto2.pressure, pvto2.co, fill_value="extrapolate")
        fvf_pb = interp1d(pvto2.pressure, pvto2.fvf, fill_value="extrapolate")

        fig, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 7))
        ax1.plot(pvto2.pressure, pvto2.fvf, 'o-', color='tab:blue', \
                 label='saturated (PVTO)')
        for n,i in enumerate(pvto.ind1.unique()):
            temp = pvto.loc[pvto.ind1 == i, :].copy()
            pb, fvfb = temp.pressure.iloc[0], temp.fvf.iloc[0]
            co = co_of_pb(pb)
            fvfb = fvf_pb(pb)
            fvf_temp1 = fvfb*np.exp(co*(pb-temp.pressure))
            if n==0:
                ax1.plot(temp.pressure, temp.fvf, '.', color='tab:blue',\
                         label='undersaturated (PVTO)') 
                ax1.plot(temp.pressure, fvf_temp1, '--', color='cyan',\
                         label='undersaturated (approximated)')
            else:
                ax1.plot(temp.pressure, temp.fvf, '.', color='tab:blue')
                ax1.plot(temp.pressure, fvf_temp1, '--', color='cyan')
            
        ax1.grid()
        ax1.set_xlabel('pressure, bar')
        ax1.set_ylabel('FVF, rm3/sm3', color='tab:blue')
        ax1.set_title('oil FVF and Rs')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(pvto2.pressure, pvto2.rs, '.-', color='tab:red')
        ax2.set_ylabel('Rs, sm3/sm3', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax1.spines['left'].set_color('tab:blue')
        ax1.spines['right'].set_color('tab:red')
        ax2.spines['left'].set_color('tab:blue')
        ax2.spines['right'].set_color('tab:red')


    pvto2.rename(columns={'pressure': 'pressure (bar)',
                          'rs': 'Rs (sm3/sm3)',
                          'fvf': 'FVF (rm3/sm3)',
                          'co': 'co (1/bar)'}, inplace=True)

    co_table = pvto2[['pressure (bar)', 'co (1/bar)']]
    rs_table = pvto2[['pressure (bar)', 'Rs (sm3/sm3)']]
    fvf_table = pvto2[['pressure (bar)', 'FVF (rm3/sm3)']]

    return fvf_table, rs_table, co_table, pvto0

def read_pvtg(p, units='metric'):
    '''
    fetch gas properties from Eclipse PVTG keyword
    NB! Only the first table can be read!
    NB! No "1*"-style entries! 
    NB! FVF is retrieved only for the first VGOR entry for each pressure node

    Parameters
    ----------
    p : str
        path to a file with PVTG keyword
    units : str, 'metric'|'field'
        field units are converted to metric

    Returns
    ----------
    fvf_table : pd.DataFrame
        FVF (2nd column, rm3/sm3) vs. pressure (1st column, bar)
    pvtg: pd.DataFrame
        raw PVTG data    
    '''
    
    ind1, ind2 = [], []
    pr, vogr, bg, visc = [], [], [], []
    # VGOR = vapor oil-gas ratio

    i1 = 0  # block index
    i2 = 0  # line index within a block

    f = open(p, 'r')
    for line in f:
        a = line.split()
        if len(a) > 0 and a[0] == 'PVTG': break 

    if len(a) > 0 and a[0] == 'PVTG':
        # reading blocks
        a = f.readline().split()
        # to pass empty strings and '--'
        while (len(a)==0) or (a[0][:2]=='--'):
            a = f.readline().split()

        c, c_max = 0, 10000
        
        while c < c_max:
            c += 1            
            # to remove '--abcdef sd' from lines like 'x y z --abcdef sd'
            for n,w in enumerate(a):
                if w[:2] == '--': 
                    a = a[:n]
                    break

            if a[0] == '/': break

            if a[-1] == '/':
                if len(a) == 5:
                    pr.append(eval(a[0]))
                    vogr.append(eval(a[1]))                    
                    bg.append(eval(a[2]))
                    visc.append(eval(a[3]))
                else:
                    pr.append(pr[-1])
                    vogr.append(eval(a[0]))                    
                    bg.append(eval(a[1]))
                    visc.append(eval(a[2]))
                ind1.append(i1)
                ind2.append(i2)
                i1 += 1
                i2 = 0
            else:
                if len(a) == 4:
                    pr.append(eval(a[0]))
                    vogr.append(eval(a[1]))                    
                    bg.append(eval(a[2]))
                    visc.append(eval(a[3]))
                else:
                    pr.append(pr[-1])
                    vogr.append(eval(a[0]))                    
                    bg.append(eval(a[1]))
                    visc.append(eval(a[2]))
                ind1.append(i1)
                ind2.append(i2)
                i2 += 1
            a = f.readline().split()
            # to pass empty strings and '--'
            while (len(a)==0) or (a[0][:2]=='--'):
                a = f.readline().split()

        f.close()
    else:
        f.close()
        raise Exception(f'PVTG keyword was not found in {p}')
        # return fvf_table, rs_table, co_table, pvtg0

    # dataframe with all the points
    pvtg = pd.DataFrame(data={'ind1': ind1, 'ind2': ind2,
                        'vogr': vogr, 'pressure': pr, 'fvf': bg, 'visc': visc})
    pvtg0 = pvtg.copy()

    if units.lower() == 'field':
        cf_ogr = 1/1.801175e-1  # from bbl/cf to m3/m3
        cf_psi2bar = 6.894757e-02
        pvtg['pressure'] = pvtg['pressure']*cf_psi2bar
        pvtg['vogr'] = pvtg['vogr']*cf_ogr*1000

    # dataframe with only first VOGR entry for each pressure node
    pvtg2 = pvtg.loc[pvtg.ind2 == 0, :].copy()
    pvtg2.set_index('ind1', inplace=True)
    pvtg2.drop('ind2', axis=1, inplace=True)

    pvtg2.rename(columns={'pressure': 'pressure (bar)',
                          'vogr': 'VOGR (sm3/sm3)',
                          'fvf': 'FVF (rm3/sm3)',
                          'co': 'co (1/bar)'}, inplace=True)

    fvf_table = pvtg2[['pressure (bar)', 'FVF (rm3/sm3)']]

    return fvf_table, pvtg0

def run_read_e100(p, reload=True, eclrun_path=r"C:\ecl\macros\eclrun.exe", \
                  version='latest', description=None, print_log=True):
    '''run Eclipse (e100) model and read results from RSM file

    Parameters
    ----------    
    p : str
        path to a DATA file
    reload: bool, optional
        reload or not saved results from a pkl file if possible.
        If reload==True and RSM file already exists => run is skipped
    eclrun_path : str, optional
        path to eclrun.exe
    version : str, optional
        eclipse version
    description : str, dict, list etc., optional
        case description to be passed to INFO (see below)
    print_log : bool, optional
        print log or not         

    Returns
    -------
    DF : pd.DataFrame with Eclipse results
    units : dict with units
    info : dict with meta data: file path, name and header
    '''
    rsm = p.replace('.DATA', '.RSM')
    rsm = rsm.replace('.data', '.RSM')
    if reload == True and os.path.exists(rsm):
        pass
    else:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

        if version=='latest':
            l = f'{eclrun_path} eclipse "{p}"'
        else:
            l = f'{eclrun_path} --version={version} eclipse "{p}"'
        if print_log: print(l)
        a = os.system(l)
        if a == -1:
            warn('\n   {l} returned some error! Check the paths and version!')
        else:
            if print_log: print(p + ' - finished')
    DF, units, info = read_rsm(rsm, reload, description, print_log)
    return DF, units, info


def run_read_e300(p, reload=True, \
    eclrun_path=r"C:\ecl\macros\eclrun.exe", 
    version = 'latest', description=None, print_log=True):
    
    '''run e300 model and read results from RSM file

    Parameters
    ----------    
    p : str
        path to a DATA file
    reload: bool, optional
        reload or not saved results from a pkl file if possible.
        If reload==True and RSM file already exists => run is skipped
    eclrun_path : str, optional
        path to eclrun.exe
    version : str, optional
        eclipse version
    description : str, dict, list etc., optional
        case description to be passed to INFO (see below)
    print_log : bool, optional
        print log or not          

    Returns
    -------
    DF : pd.DataFrame with Eclipse results
    units : dict with units
    info : dict with meta data: file path, name and header
    '''
    rsm = p.replace('.DATA', '.RSM')
    rsm = rsm.replace('.data', '.RSM')
    if reload == True and os.path.exists(rsm):
        pass
    else:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
                
        if version=='latest':
            l = f'{eclrun_path} e300 "{p}"'
        else:
            l = f'{eclrun_path} --version={version} e300 "{p}"'
        if print_log: print(l)
        a = os.system(l)
        if a == -1:
            warn('\n   {l} returned some error! Check the paths and version!')
        else:
            if print_log: print(p + ' - finished')
    DF, units, info = read_rsm(rsm, reload, description, print_log)
    return DF, units, info
