import xhermes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import sys

#----- Load in the shot data and input options using xHermes
# filePath = sys.argv[1]
# ds1       = xhermes.open(filePath,keep_yboundaries=True) # Load in with all guard cells
# ds1       = ds1.hermes.extract_1d_tokamak_geometry()      # Extract and normalise outputs
# ds1       = ds1.isel(pos=slice(1,-1))                     # Crop off outer guard cells, keep inners for target values
# ds1["t"]  = ds1["t"]*1e3                                  # Convert t into ms from s
# guardRep = False

#----- Load in the shot data and input options using sdtools
sys.path.append("/home/mbk513/sdtools")
from hermes3.load import *
filePath  = sys.argv[1]
ds2       = Load.case_1D(filePath).ds                     # Load, extract, normalise, and guard replace outputs
ds2       = ds2.isel(pos=slice(1,-1))                     # Crop off outer guard cells, keep inners for target values
ds2["t"]  = ds2["t"]*1e3                                  # Convert t into ms from s
guardRep = True

ds = ds2

ds["dv"] = ds["J"]*ds["dx"]*ds["dy"]*ds["dz"]

# def tanh(x,x0):
#     return (np.exp(2.*(x-x0))-1.)/(np.exp(2.*(x-x0))+1.)

# #----- Source up to x-point
# sourceVolume = ds["dv"].values[1:-1]
# sourceVolume[np.argmin(np.abs(ds["pos"].values-50.)):] = 0
# sourceVolume = np.trapz(sourceVolume,ds["pos"].values[1:-1])
#
# #----- Tanh Pump
# pumpRegion     = (0.5*tanh(ds["pos"].values[1:-1],62.5)+0.5)*(-0.5*tanh(ds["pos"].values[1:-1],87.5)+0.5)
# # pumpRegion[:np.argmin(np.abs(ds["pos"].values-0.))] = 0.
# pumpVolumeTanh = np.trapz(ds["dv"].values[1:-1]*pumpRegion,ds["pos"].values[1:-1])

#----- Load in calculate sheath gammas, cross-sectional area, and elementary charge
options = BoutData(filePath, yguards = True, info = False, strict = True, DataFileCaching=False)["options"]
try:
    gamma_e = float(options["sheath_boundary_simple"]["gamma_e"])
    gamma_i = float(options["sheath_boundary_simple"]["gamma_i"])
except:
    gamma_e = 4.5
    gamma_i = 2.5
csArea  = ds["dx"].values*ds["dz"].values*ds["J"].values/np.sqrt(ds["g_22"].values)
q_e     = 1.602e-19

#----- Calculate the target heat flux from the inner guard cells and the final domain cell
if(guardRep):
    Q_t_electrons = gamma_e*ds["Ne"][:,-1] *q_e*ds["Te"][:,-1] *ds["Ve"][:,-1] *csArea[-1]#ds["J"]/np.sqrt(ds["g_22"])#ds["da"][0,-1]
    Q_t_ions      = gamma_i*ds["Nd+"][:,-1]*q_e*ds["Td+"][:,-1]*ds["Vd+"][:,-1]*csArea[-1]#ds["J"]/np.sqrt(ds["g_22"])#ds["da"][0,-1]
    #----- Units  = [m^-3]*[J]*[m.s^-1]*[m^2] = J.s^-1 = W
else:
    Q_t_electrons = gamma_e*(ds["Ne"][:,-1] +ds["Ne"][:,-2]) *0.5*q_e*(ds["Te"][:,-1] +ds["Te"][:,-2]) *0.5*(ds["Ve"][:,-1] +ds["Ve"][:,-2]) *0.5*csArea[-1]
    Q_t_ions      = gamma_i*(ds["Nd+"][:,-1]+ds["Nd+"][:,-2])*0.5*q_e*(ds["Td+"][:,-1]+ds["Td+"][:,-2])*0.5*(ds["Vd+"][:,-1]+ds["Vd+"][:,-2])*0.5*csArea[-1]
Q_t = Q_t_electrons+Q_t_ions

#----- Calculate the target particle flux from the inner guard cells and the final domain cell
if(guardRep):
    G_t = 3.0*ds["Nd+"][:,-1]*q_e*ds["Vd+"][:,-1]*csArea[-1]#ds["J"]/np.sqrt(ds["g_22"])#ds["da"][0,-1]
    #----- Units  = [m^-3]*[J]*[m.s^-1]*[m^2] = J.s^-1 = W
else:
    G_t = 3.0*(ds["Nd+"][:,-1]+ds["Nd+"][:,-2])*0.5*q_e*0.5*(ds["Vd+"][:,-1]+ds["Vd+"][:,-2])*0.5*csArea[-1]

#----- Crop off inner guard cells or the boundary values as they are no longer needed
ds = ds.isel(pos=slice(1,-1))

#----- Old calculations for the total energy contained within each species
# totalElectronEng = (3./2.)*(ds["Pe"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalIonEng      = (3./2.)*(ds["Pd+"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalNeutralEng  = (3./2.)*(ds["Pd"]*ds["dv"]).sum("pos")#* (ds["t"] - ds["t"][0])
# totalSystemEng   = totalElectronEng+totalIonEng+totalNeutralEng

#----- Total Input Power
electronInput   = (3./2.)*(ds["Pe_src"]*ds["dv"]).sum("pos")# Pe_src is in Pa.s^-1 which is J.m^-3.s^-1 = W.m^-3
ionInput        = (3./2.)*(ds["Pd+_src"]*ds["dv"]).sum("pos")
#----- Impurity Radiation
totalRadArPow   = (ds["Rar"]*ds["dv"]).sum("pos")
#----- Hydrogenic Radiation
totalRadHexPow  = np.abs((ds["Rd+_ex"] *ds["dv"]).sum("pos")) # Needs to be negative because of output conventions
totalRadHrecPow = np.abs((ds["Rd+_rec"]*ds["dv"]).sum("pos"))
#----- CX
totalCXKinetic  = 0.5*((abs(ds["Fdd+_cx"])*abs(ds["Vd+"]))*ds["dv"]).sum("pos") # E = 0.5* mv*v ???
totalCXTherm    = (ds["Edd+_cx"]*ds["dv"]).sum("pos")
#----- Energy Source from dissociating neutral molecules (from recycling)
totalDissNeut  = G_t

#----- Total Power Loss
totalPowLoss    = Q_t+totalRadArPow+totalRadHexPow+totalRadHrecPow
#----- Total Input Power
totalInputPow   = electronInput+ionInput
totalPowsrc     = totalInputPow+totalDissNeut
#----- Power Imbalance
powerImbalance = totalPowsrc.values[-1]-totalPowLoss.values[-1]
powerImbPerc   = 100.*(totalPowsrc.values[-1]-totalPowLoss.values[-1])/totalPowsrc.values[-1]

#----- Total energy needed to ionise all the neutrals
totalIoniseEng  = (ds["Nd"]*ds["dv"]*13.6*q_e).sum("pos")
powIncFactor    = float(options["Pe"]["powFactor"])
deltaPower      = (powIncFactor-1.)*(electronInput+ionInput).values[0]
timeToIonise    = totalIoniseEng.values[0]/deltaPower # J/(J/s) = s
print("Input power = %.3e W, total energy needed to ionise neutrals %.3e J"%(totalPowsrc.values[-1],totalIoniseEng.values[0]))
print("Delta Power = %.3e W (factor %.3fx increase), time needed (for zero additional losses) = %.3e ms"%(deltaPower,powIncFactor,timeToIonise*1e3))
# sys.exit()

print("\nSources")
print("-----")
print("{:<20} {:.3e}".format("Electron Source: ",electronInput.values[-1]))
print("{:<20} {:.3e}".format("Ion Source: ",ionInput.values[-1]))
print("{:<20} {:.3e}".format("Recycled Neutrals: ",G_t.values[-1]))

print("\nLosses")
print("-----")
print("{:<20} {:.3e}".format("Excitation: ",totalRadHexPow.values[-1]))
print("{:<20} {:.3e}".format("Recombination: ",totalRadHrecPow.values[-1]))
print("{:<20} {:.3e}".format("Electorn sheath: ",Q_t_electrons.values[-1]))
print("{:<20} {:.3e}".format("Ion sheath: ",Q_t_ions.values[-1]))

print("\nPower balance")
print("-----")
print("{:<20} {:.3e}".format("Total Source: ",totalPowsrc.values[-1]))
print("{:<20} {:.3e}".format("Total Losses: ",totalPowLoss.values[-1]))
print("{:<20} {:.3e} / {:.1f}%".format("Power Imbalance: ",powerImbalance,powerImbPerc))

#----- Sources
plt.plot(ds["t"].values,totalPowsrc.values,    label="Total Source", color="firebrick")
plt.plot(ds["t"].values,totalInputPow.values,  label="Input Power",  color="tab:orange",linestyle="--")
plt.plot(ds["t"].values,totalDissNeut,         label="Nuetral Diss.",color="crimson",   linestyle="--")
#----- Sinks
plt.plot(ds["t"].values,totalPowLoss.values,   label="Total Losses",       color="midnightblue")
plt.plot(ds["t"].values,totalRadArPow.values,  label="Ar. Rad.",           color="mediumblue")
plt.plot(ds["t"].values,totalRadHexPow.values, label="H.Exc. Rad.",        color="mediumpurple")
plt.plot(ds["t"].values,totalRadHrecPow.values,label="H.Rec. Rad",         color="darkorchid")
plt.plot(ds["t"].values,Q_t_ions,              label="Ion sheath trans.",  color="tab:purple")
plt.plot(ds["t"].values,Q_t_electrons,         label="Elec. sheath trans.",color="royalblue")
# plt.ylim([-0.1e9,1.1e9])
plt.legend(loc="best",ncol=2)
plt.xlabel("Time (ms)")
plt.ylabel("Power (W)")
plt.show()
