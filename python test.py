import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------------------------------------
# NIST XCOM mass attenuation coefficients for Aluminum
# μ/ρ values from:
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html
#
# Energy in keV, μ/ρ in cm^2/g
# -------------------------------------------------------

nist_energy_keV = np.array([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0
])

nist_mu_over_rho = np.array([
    1185.0,
    226.0,
    78.8,
    35.0,
    19.34,
    11.53,
    5.033,
    2.623,
    0.7955,
    0.3441
])

# Interpolation function μ/ρ(E)
mu_interp = interp1d(
    nist_energy_keV,
    nist_mu_over_rho,
    kind="cubic",
    fill_value="extrapolate"
)

# Aluminum density
rho_al = 2.70  # g/cm^3

def transmission_al(thickness_inch, energies_keV):
    """
    Compute transmission through aluminum.

    thickness_inch : thickness in inches
    energies_keV    : array of photon energies (keV)
    """
    thickness_cm = thickness_inch * 2.54

    mu_over_rho = mu_interp(energies_keV)     # cm^2/g
    mu_linear = mu_over_rho * rho_al          # cm^-1

    return np.exp(-mu_linear * thickness_cm)

# -------------------------------------------------------
# Compute transmission curves
# -------------------------------------------------------

energies = np.linspace(1, 20, 500)

T_002 = transmission_al(0.002, energies)
T_007 = transmission_al(0.007, energies)

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(energies, T_002, label="0.002 inch Al")
plt.plot(energies, T_007, label="0.007 inch Al")

plt.yscale("log")
plt.xlabel("Energy (keV)")
plt.ylabel("Transmission")
plt.title("Aluminum Transmission Curves (NIST XCOM)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Mark Fe-55 line
plt.axvline(5.9, linestyle=":", label="Fe-55 (5.9 keV)")

plt.legend()
plt.show()

# -------------------------------------------------------
# Print transmission at Fe-55 energy
# -------------------------------------------------------

E_fe = 5.9
print("Transmission at 5.9 keV:")
print(f"  0.002 inch Al: {transmission_al(0.002, np.array([E_fe]))[0]:.4f}")
print(f"  0.007 inch Al: {transmission_al(0.007, np.array([E_fe]))[0]:.6f}")
