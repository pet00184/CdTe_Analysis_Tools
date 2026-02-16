import response_tools.io.fetch_response_data as fetch
fetch.foxsi4_download_required(verbose=True)

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import response_tools.attenuation as att
import response_tools.responses as responses

# to get energy arrays
region = 0 #all regions checked and much the same
t2_rmf = responses.foxsi4_telescope2_rmf(region=region)
photon_mid_energies = (t2_rmf.input_energy_edges[:-1]+t2_rmf.input_energy_edges[1:])/2
count_mid_energies = (t2_rmf.output_energy_edges[:-1]+t2_rmf.output_energy_edges[1:])/2

# source activity in Berkeley 2025
# bought: 15 July 2012
# half-life: ~2.7 years
# Initial activity: 3.7 MBq
# so over 13.25 years: (1/2)**(13.24/2.7) = 0.03332139458380741
# so at previous Berkeley integration: (1/2)**((13.25-1.67)/2.7) = 0.05115826372062386
# so since previous Berkeley integration: (1/2)**(1.67/2.7) = 0.6513394349303196
activity = 111e3 << u.Bq

#
# How is the activity measured? Is this the total activity that is isotrpoic?
# Should normalised surface hitting the detector area
#

_line = norm.pdf(photon_mid_energies, 5.9<<u.keV, 0.1<<u.keV)
fe_line = _line/np.sum(_line) * activity.value<<u.ph/u.second
print(f"Sum of all photons/second should equal {activity}:", np.sum(fe_line))

plt.figure()
plt.plot(photon_mid_energies, fe_line)
plt.ylabel(f"Photons [{fe_line.unit:latex}]")
plt.xlabel(f"Photon Energy [{photon_mid_energies.unit:latex}]")
plt.title("Fe Line")
plt.show()

pix = att.att_pixelated(photon_mid_energies, use_model=True)
mylar = att.att_al_mylar(photon_mid_energies)

p2_att = att.att_uniform_al_cdte(photon_mid_energies, position=2) # pos 2 381 µm Al
p4_att = att.att_uniform_al_cdte(photon_mid_energies, position=4) # pos 4 127 µm Al

print(pix.transmissions, p2_att.transmissions, p4_att.transmissions)

# Nitrogen gas has attenuation at 6 keV:
# [1] https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z07.html
# [2] https://www.nde-ed.org/Physics/X-Ray/attenuationCoef.xhtml
# [3] https://www.onsitegas.com/blog/density-of-nitrogen-gas/
# Say from source-to-det is 3 cm in total
# >>> np.e**(-3*1.809e1*0.0012506)
# 0.9343818774019435
n2 = 0.9343818774019435

pix_passthrough = pix.transmissions*mylar.transmissions*fe_line*n2
p2_att_passthrough = p2_att.transmissions*fe_line*n2
p4_att_passthrough = p4_att.transmissions*fe_line*n2
print(pix_passthrough, p2_att_passthrough, p4_att_passthrough)

# %%
# What does the count spectrum look like
# --------------------------------------
#
# The other Fe55 line is ignore and does not eat up any of the photon 
# signal either.
#
# Get one photon/Bq, assume region 0
t3_rmf = responses.foxsi4_telescope3_rmf(region=region)
t4_rmf = responses.foxsi4_telescope4_rmf(region=region)
t5_rmf = responses.foxsi4_telescope5_rmf(region=region)

t2_rmf.response[np.isnan(t2_rmf.response)] = 0
t3_rmf.response[np.isnan(t3_rmf.response)] = 0
t4_rmf.response[np.isnan(t4_rmf.response)] = 0
t5_rmf.response[np.isnan(t5_rmf.response)] = 0

counts2 = (p2_att_passthrough.value<<u.ph/u.second) @ t2_rmf.response
counts3 = (pix_passthrough.value<<u.ph/u.second) @ t3_rmf.response
counts4 = (p4_att_passthrough.value<<u.ph/u.second) @ t4_rmf.response
counts5 = (pix_passthrough.value<<u.ph/u.second) @ t5_rmf.response

plt.figure()
plt.errorbar(count_mid_energies, counts2, yerr=np.sqrt(counts2.value)<<u.ct/u.second, ls="", marker="+", label="pos.2 (381 µm Al)")
plt.errorbar(count_mid_energies, counts3, yerr=np.sqrt(counts3.value)<<u.ct/u.second, ls="", marker="+", label="pos.3 (pix.+Mylar)")
plt.errorbar(count_mid_energies, counts4, yerr=np.sqrt(counts4.value)<<u.ct/u.second, ls="", marker="+", label="pos.4 (127 µm Al)")
plt.errorbar(count_mid_energies, counts5, yerr=np.sqrt(counts5.value)<<u.ct/u.second, ls="", marker="+", label="pos.5 (pix.+Mylar)")
plt.axhline(20, color="k", label="~20-30 cts/s min., no source, at -20 C")
plt.axhline(30, color="k")
plt.legend()
plt.xlim([4,10])
plt.ylim([2e-1,230])
plt.xlabel(f"Count Energy [{count_mid_energies.unit:latex}]")
plt.ylabel(f"Count Rate [{counts2.unit:latex}]")
plt.title(f"Fe Line")
plt.yscale("log")
plt.show()

# %%
# How many counts don't make it because the noise make the trigger be 
# tagged as a multi-clump event?
#
# Only one detector from FOXSI-4 stands a chance of seeing the Fe-line 
# and none of them saw it through Lindsay and Yixian's testing.
#
# Only the most weakly attenuated FOXSI-5 CdTe detector managed to see
# the line, the other fixed attenuated detector has a thickness 7/5ths 
# more than position 4 in FOXSI-4.
#
# Nitrogen gas has attenuation at 6 keV:
# [1] https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z07.html
# [2] https://www.nde-ed.org/Physics/X-Ray/attenuationCoef.xhtml
# [3] https://www.onsitegas.com/blog/density-of-nitrogen-gas/
# >>> np.e**(-3*1.809e1*0.0012506)
# 0.9343818774019435