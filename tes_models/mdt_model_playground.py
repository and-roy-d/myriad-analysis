import numpy as np
import pylab as plt
import scipy
import lmfit
from lmfit import minimize, Parameters
from mdt3 import mdt3_core
from mdt3 import tes_simple, tes_intervening, tes_dangling


tes = mdt3_core.MDT3_Core()
tsimp = tes_simple.TES_Simple(tes)
tdang = tes_dangling.TES_Dangling(tes)
tint = tes_intervening.TES_Intervening(tes)


tes_params = tes.makeDefaultParamsDict(num_sets=1)
tes_params2 = tes.makeDefaultInterParamsDict(num_sets=1)
print(tes_params2)



Tc = 0.053
absorber_size = 250**2

Tc_scaling = Tc/0.053
absorber_scaling = absorber_size/250**2

delE_scaling = 0.9/0.65  # measured FWHM at 300 eV is 0.9 eV

thermal_n = 3.8
E_photon_eV = 2.4

tes_params['t_0'].set(value=0.00316)
tes_params['f_0'].set(value=62500)
tes_params['L_0'].set(value=80)
tes_params['C_tes_0'].set(value=0.161*Tc_scaling*absorber_scaling)
tes_params['C_int_0'].set(value=0.161*Tc_scaling*absorber_scaling)
tes_params['C_abs_0'].set(value=0.05*Tc_scaling*absorber_scaling)

tes_params['G_tes_bath_0'].set(value=0.093*Tc_scaling**(thermal_n -1))
tes_params['G_abs_tes_0'].set(value=0.655*Tc_scaling**(thermal_n -1))
tes_params['R_0_0'].set(value=0.00047)
tes_params['R_L_0'].set(value=0.00025)
tes_params['alpha_I_0'].set(value=1365)
tes_params['beta_I_0'].set(value=64)
tes_params['T_tes_0'].set(value=0.053*Tc_scaling)
tes_params['T_bath_0'].set(value=0.021)
tes_params['initE_0'].set(value=E_photon_eV)
tes_params['n_mem_0'].set(value=thermal_n)
tes_params['M_0'].set(value=2.2)


tes_params2['t_0'].set(value=0.00316)
tes_params2['f_0'].set(value=62500)
tes_params2['L_0'].set(value=120)
tes_params2['C_tes_0'].set(value=0.05*Tc_scaling*absorber_scaling)
tes_params2['C_int_0'].set(value=0.161*Tc_scaling*absorber_scaling)

tes_params2['G_tes_bath_0'].set(value=0.093*Tc_scaling**(thermal_n -1))
tes_params2['G_tes_int_0'].set(value=0.65*Tc_scaling**(thermal_n -1))
tes_params2['R_0_0'].set(value=0.00047)
tes_params2['R_L_0'].set(value=0.00025)
tes_params2['alpha_I_0'].set(value=600)
tes_params2['beta_I_0'].set(value=15)
tes_params2['T_tes_0'].set(value=0.053*Tc_scaling)
tes_params2['T_bath_0'].set(value=0.021)
tes_params2['initE_0'].set(value=E_photon_eV)
tes_params2['n_mem_0'].set(value=thermal_n)
tes_params2['M_0'].set(value=2.2)

print(tes_params.valuesdict())

lambda1_d, lambda2_d, lambda3_d, a1_d, a2_d, a3_d, *others = tdang.calc_lambda(tes_params)
print('Dangling:',lambda1_d, lambda2_d, lambda3_d)
print('Dangling:',a1_d, a2_d, a3_d)
lambdaplus, lambdaminus = tsimp.calc_lambda(tes_params)
print('Simple:', lambdaplus,lambdaminus)

t_vals = np.linspace(0.0,0.002,1001)
Ites_vals_d, Ttes_vals_d, Tabs_vals_d = tdang.calc_pulse(tes_params, t_vals)
# Ites_vals_i, Ttes_vals_i, Tabs_vals_i = tint.calc_pulse(tes_params2, t_vals)

plt.figure()
plt.plot(t_vals*1e3, Ites_vals_d,'-',label='dangling')
# plt.plot(t_vals, Ites_vals_i,'-',label='intervening')
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('TES Current (A)')



plt.figure()

plt.plot(t_vals*1e3, Ttes_vals_d*1e6,'-',label='Ttes dangling')
plt.plot(t_vals*1e3, Tabs_vals_d*1e6,'-',label='Tabs dangling')
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (uK)')
plt.legend()
print('Tc scaling', Tc_scaling)
print('Absorber size scaling', absorber_scaling)
print('\n************************************************************************\n')
print('Dangling model FWHM: ',tdang.calc_energy_res(tes_params), ' eV')
print('Predicted energy resolution at 300 eV:', tdang.calc_energy_res(tes_params)*delE_scaling, ' eV')

print('\n************************************************************************\n')
print('Simple model FWHM: ',tsimp.calc_energy_res(tes_params), ' eV')
print('Predicted energy resolution at 300 eV:', tsimp.calc_energy_res(tes_params)*delE_scaling, ' eV')


plt.show()
