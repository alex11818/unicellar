# unicellar 0.1.0. Single-cell reservoir modeling toolbox
Python library to quickly set up and run single-cell reservoir models of CO2/H2 storages and petroleum fields

Single-cell reservoir model (SCRM) = material balance equation (MBE) + analytical aquifer  
SCRM is solved for pressure at each time step to forecast pressure dynamics for known fluid production/injection scenarios.  
Alternatively, MBE can be solved for ultimate storage capacity (USC) to get an estimate of the maximal fluid volume that can be stored in the reservoir at a given pressure and cumulative production/injection.

## Use cases:
1. to history-match the reservoir pressure dynamics by tuning reservoir and aquifer parameters
2. to quickly estimate the Ultimate Storage Capacity (USC) of the reservoir
for different reservoir parameters and storage scenarios.
3. to forecast reservoir pressure dynamics for a production/injection scenario
4. to quickly set up a fast proxy model to support a full-field model's design and history matching 

## OK. What else can it do?
It also features functions:
- to plot results in nice & neat Plotly charts 
- to read Eclipse results from RSM files
- to read Eclipse (E100) PVT keywords (PVTO, PVCO, PVTG, PVDG) to set up fluid 
properties
- to fetch various PVT properties from the NIST database [1] 
- to estimate the work required to compress a fluid from one pressure to another
- to estimate bottomhole pressure (BHP) from tubing head pressure (THP) and vice versa through the column weight

# Installation  
1. Download or clone the library. 
2. Navigate to the root folder containing setup.py by `cd`
3. run `pip install .`  
(worked fine with Python 3.5/3.7/3.8)

Unicellar is built on numpy, scipy, pandas, matplotlib, and plotly. 
So there are no exotic/esoteric dependencies, and you are almost certain to meet all the requirements. 
Nevertheless, you may create a test environment in Anaconda package manager by:
```
conda create -n uTest python=3.8
conda activate -n uTest
```

# What to start with?
Check out \examples (arranged by complexity):
* radial_aquifer.py
* gas_storage.py
* single-cells.py
* oil+gas_cap+aquifer.py
* lbr1.py

## On the model:
A black oil formulation is employed for PVT props. is largely compatible with black oil reservoir simulators (like Eclipse, OPM, tNavigator etc.). 
The following components are currently available:
- oil (with dissolved gas)
- free gas that can dissolve in oil
- water
- CO2: an inert storage fluid that does not dissolve in other fluids. It can be CO2, H2, CH4 etc. 

## On the name choice:
UNICELLAR = unicellular (an organism that consists of a single cell) + 
cellar (an underground storage room)

# Acknowledgements
This code has been developed in the REPP-CO2, STRATEGY CCUS, and CO2-SPICER projects.
The REPP-CO2 project was supported by Norway Grants from the CZ-08 Carbon Capture and Storage programme (Norway Grants 2009-2014). The STRATEGY CCUS project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 837754. The CO2-SPICER project benefits from a € 2.32 mil. grant from Norway and Technology Agency of the Czech Republic. 


# References
1. Eric W. Lemmon, Ian H. Bell, Marcia L. Huber, and Mark O. McLinden, 
    "Thermophysical Properties of Fluid Systems"  in NIST Chemistry WebBook, 
    NIST Standard Reference Database Number 69, 
    Eds. P.J. Linstrom and W.G. Mallard, National Institute of Standards and Technology, 
    Gaithersburg MD, 20899, https://doi.org/10.18434/T4D303, (retrieved November 29, 2022).

2. Hladik V. et al., "LBr-1 – Research CO2 Storage Pilot in the Czech Republic", 
    DOI:10.1016/j.egypro.2017.03.171
