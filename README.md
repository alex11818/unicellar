# Unicellar 0.1.0: Single-Cell Reservoir Modeling Toolbox

**Unicellar** is a Python library designed to quickly set up and run single-cell reservoir models (SCRM) for CO2/H2 storage and petroleum fields.

## Overview
A Single-Cell Reservoir Model (SCRM) combines a Material Balance Equation (MBE) with an analytical aquifer. The model is solved for pressure at each time step to forecast pressure dynamics for known fluid production/injection scenarios. Alternatively, MBE can be solved for ultimate storage capacity (USC) to estimate the maximal fluid volume that can be stored in the reservoir at a given pressure and cumulative production/injection.

## Use Cases
1. **History Matching:** Tune reservoir and aquifer parameters to match the reservoir pressure dynamics.
2. **Estimate Ultimate Storage Capacity (USC):** Quickly estimate the USC of the reservoir for different parameters and storage scenarios.
3. **Forecast Pressure Dynamics:** Predict reservoir pressure dynamics for a given production/injection scenario.
4. **Fast Proxy Model:** Set up a fast proxy model to support full-field model design and history matching.

![Figure 2](https://github.com/alex11818/unicellar/assets/53487462/ef8c17de-7f41-4721-8461-ae14df87b4e6)

## Features
- Plot results in neat Plotly charts.
- Read Eclipse results from RSM files.
- Read Eclipse (E100) PVT keywords (PVTO, PVCO, PVTG, PVDG) to set up fluid properties.
- Fetch various PVT properties from the NIST database.
- Estimate the work required to compress a fluid from one pressure to another.
- Estimate bottomhole pressure (BHP) from tubing head pressure (THP) and vice versa through the column weight.

## Installation
### Using pip
1. Download or clone the library.
2. Navigate to the root folder containing `setup.py`:
3. Install the library:

### Using Anaconda
1. Create a test environment in Anaconda:
conda create -n uTest python=3.8
conda activate uTest
2. Navigate to the root folder containing `setup.py`:
cd path/to/unicellar
3. Install the library:
pip install .


*Unicellar is built on numpy, scipy, pandas, matplotlib, and plotly. These dependencies are common and should be readily available.*

## Getting Started
Check out the examples in the `/examples` folder, arranged by complexity:
- `radial_aquifer.py`
- `gas_storage.py`
- `single-cells.py`
- `oil+gas_cap+aquifer.py`
- `lbr1.py`

## Model Information
The black oil formulation used for PVT properties is largely compatible with black oil reservoir simulators (like Eclipse, OPM, tNavigator, etc.). The following components are currently available:
- Oil (with dissolved gas)
- Free gas that can dissolve in oil
- Water
- CO2 (an inert storage fluid that does not dissolve in other fluids, which can be CO2, H2, CH4, etc.)

## Name Origin
**UNICELLAR** = unicellular (an organism that consists of a single cell) + cellar (an underground storage room)

## Acknowledgements
This code has been developed in the REPP-CO2, STRATEGY CCUS, and CO2-SPICER projects.
- The REPP-CO2 project was supported by Norway Grants from the CZ-08 Carbon Capture and Storage programme (Norway Grants 2009-2014).
- The STRATEGY CCUS project received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 837754.
- The CO2-SPICER project benefits from a €2.32 mil. grant from Norway and the Technology Agency of the Czech Republic.

## References
1. Eric W. Lemmon, Ian H. Bell, Marcia L. Huber, and Mark O. McLinden, "Thermophysical Properties of Fluid Systems" in NIST Chemistry WebBook, NIST Standard Reference Database Number 69, Eds. P.J. Linstrom and W.G. Mallard, National Institute of Standards and Technology, Gaithersburg MD, 20899, [Link](https://doi.org/10.18434/T4D303), (retrieved November 29, 2022).

2. Hladik V. et al., "LBr-1 – Research CO2 Storage Pilot in the Czech Republic", 13th International Conference on Greenhouse Gas Control Technologies, GHGT-13, 14-18 November 2016, Lausanne, Switzerland, [Link](https://doi.org/10.1016/j.egypro.2017.03.1712)

3. Khrulenko, A., Berenblyum, R., "Single-cell reservoir model for CO2 storage planning", Conference poster, TCCS-12 (12th Trondheim Conference on CO2 Capture, Transport and Storage), 19-21 June 2023, Trondheim, Norway, [poster URL](https://co2-spicer.geology.cz/sites/default/files/2023-08/250_poster_Khrulenko_SCRM%20for%20CO2%20storage%20planning.pdf)
