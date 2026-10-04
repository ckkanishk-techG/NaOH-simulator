"""Fundamental physical constants and unit factors (SI). Exact/CODATA values only.

Anything that is a *modelling* number (rate constants, heat-transfer coefficients, ...) does NOT
belong here: it lives in the versioned property database (``hydra.thermo.propdb``).
"""

R = 8.314462618  # J mol-1 K-1 (exact, CODATA 2018)
F = 96485.33212  # C mol-1 (exact, CODATA 2018)
K_B = 1.380649e-23  # J K-1 (exact)
N_A = 6.02214076e23  # mol-1 (exact)
SIGMA_SB = 5.670374419e-8  # W m-2 K-4
G_ACC = 9.80665  # m s-2
P_ATM = 101325.0  # Pa
T_REF = 298.15  # K
T_STP = 273.15  # K (0 degC, STP for gas-volume reporting)
BAR = 1.0e5  # Pa
KELVIN_OFFSET = 273.15
MMHG = 133.322368  # Pa
LITRE = 1.0e-3  # m3
ML = 1.0e-6  # m3
MIN = 60.0
HOUR = 3600.0

# molar masses, kg mol-1 (IUPAC 2013 standard atomic weights)
MW_AL = 26.9815385e-3
MW_NA = 22.98976928e-3
MW_O = 15.999e-3
MW_H = 1.008e-3
MW_H2 = 2.016e-3
MW_H2O = 18.015e-3
MW_NAOH = 39.997e-3
MW_ALOH4 = MW_AL + 4 * (MW_O + MW_H)  # Al(OH)4-
MW_ALOH3 = MW_AL + 3 * (MW_O + MW_H)
MW_AIR = 28.9647e-3
