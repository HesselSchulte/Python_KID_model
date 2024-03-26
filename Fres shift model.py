#Requirements:  1000 MKIDS on single readout line 
#               The bandwidth is 4-8 GHz
#               Smallest detectable wavelength is 402 nm
#               Q factor is constant on all frequencies
#               
import numpy as np
import matplotlib.pyplot as plt
import kidcalc as kc # KID model
import SC # properties of superconducting materials

#Design parameters
N_MKIDs = 1000
N_read = int(1e3)
f0 = 4.004e9 #Hz resonance frequency
kbT0 = 86.17 * .1 #µeV, operating temperature of 100 mK
hw0 = 6.582e-4*2*np.pi*f0*1e-6 #Energy inherent in KID at fres
Qi_sat = 1e16
Qc = 1.80e4     #Coupling quality factor
V = 15 # µm^3 inductor volume
supercond = SC.bTa() # material constants
ak = .96 # Kinetic Inductance fraction
beta = 2    #Thin film beta approximation
lmbda = 402 #nm, gives the max. E for the responsivity curve
etapb = .55 #Pair breaking efficiency 

dNqp = etapb * 6.528e-4*2*np.pi* 3e8 / (lmbda * 1e-3) / supercond.D0
Nqp0 = V * kc.nqp(kbT0, supercond.D0, supercond)
kbTeff = kc.kbTeff((Nqp0 + dNqp)/V, supercond)

# calculate equilibrium complex conductivity, Qi, Lk and from that C and Lg (for later fres calc)
s10, s20 = kc.cinduct(hw0, supercond.D0, kbT0)
Qi_Nqp0 = 2/(ak*beta) * s20/s10
Qi0 = Qi_Nqp0 * Qi_sat / (Qi_Nqp0 + Qi_sat)  #Physical systems have limited Qi
Q = Qc * Qi0 / (Qc + Qi0)
Lk0 = np.imag(1/(s10-1j*s20))/(2*np.pi*f0)
C = ak/(f0**2*Lk0)  #Capacitance from resonance freq definition, factor (2pi)^2 missing?
Lg = Lk0*(1/ak-1)

f_read_min = f0 - 10*f0/Q
f_read_max = f0 + 10*f0/Q
f_read = np.linspace(f_read_min, f_read_max, N_read)
f_dis = (f_read_max-f_read_min)/N_MKIDs

#calculate complex conductivities and excess quasiparticles during pulse (between kbT0 and kbTeff)
kbTarr = np.logspace(np.log10(kbT0), np.log10(kbTeff), N_read)
s1, s2, exNqp = np.zeros((3, len(kbTarr)))
for i, kbT in enumerate(kbTarr):
    s1[i], s2[i] = kc.cinduct(hw0, supercond.D0, kbT)
    exNqp[i] = V * kc.nqp(kbT, supercond.D0, supercond)

Qi_resp = (2*s2)/(ak*beta*s1)
Q_resp = Qc * Qi_resp / (Qc + Qi_resp)
S21_0 = kc.S21(Qi0, Qc, f_read, f0)

Lk = np.imag(1/(s1-1j*s2))/(2*np.pi*f0)
f_resp = 1/np.sqrt(C*(Lk + Lg))

S21_resp = kc.S21(Qi_resp, Qc, f_read, min(f_resp))

plt.plot(f_read, np.abs(S21_0), label = "Unloaded Resonator")
plt.plot(f_read, np.abs(S21_resp), label = "highest energy incident photon")
plt.title("Resonace frequency shift")
plt.xlabel("df (GHz)")
plt.ticklabel_format(useOffset=False)
plt.ylabel("S21")
plt.legend()
plt.show()

print(min(S21_0))