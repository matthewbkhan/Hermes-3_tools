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

#----- Conversion factor for Sd+_iz or _rec
# print(ds["Sd+_iz"].attrs["conversion"])
# print(ds.attrs["metadata"]["Nnorm"])
# print(ds.attrs["metadata"]["Omega_ci"])
# print(ds.attrs["metadata"]["Nnorm"]*ds.attrs["metadata"]["Omega_ci"])
# sys.exit()

#----- Plot J
if(0):
    ds["dv"] = ds["J"]*ds["dx"]*ds["dy"]*ds["dz"]
    ds["t"]  = ds["t"]-ds["t"].values[0]
    plt.plot(ds["pos"].values,ds["J"])
    plt.show()

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
electronInput    = (3./2.)*(ds["Pe_src"]*ds["dv"])# Pe_src is in Pa.s^-1 which is J.m^-3.s^-1 = W.m^-3
ionInput         = (3./2.)*(ds["Pd+_src"]*ds["dv"])
electronInputSum = electronInput.sum("pos")
ionInputSum      = ionInput.sum("pos")
#----- Impurity Radiation
try:
    totalRadArPow    = ds["Rar"]*ds["dv"]
    totalRadArPowSum = totalRadArPow.sum("pos")
except exceptionError as Exception:
    print(exceptionError,"\nNo Ar impurity")
#----- Hydrogenic Radiation
totalRadHexPow     = ds["Rd+_ex"] *ds["dv"] # Needs to be negative because of output conventions
totalRadHrecPow    = ds["Rd+_rec"]*ds["dv"]
totalRadHexPowSum  = np.abs(totalRadHexPow.sum("pos"))
totalRadHrecPowSum = np.abs(totalRadHrecPow.sum("pos"))
#----- CX
totalCXKinetic    = 0.5*((abs(ds["Fdd+_cx"])*abs(ds["Vd+"]))*ds["dv"]) # E = 0.5* mv*v ???
totalCXTherm      = ds["Edd+_cx"]*ds["dv"]
totalCXKineticSum = totalCXKinetic.sum("pos")
totalCXThermSum   = totalCXTherm.sum("pos")
#----- Energy Source from dissociating neutral molecules (from recycling)
totalDissNeut  = G_t

#----- Total Power Loss
totalPowLoss    = Q_t+totalRadArPowSum+totalRadHexPowSum+totalRadHrecPowSum
#----- Total Input Power
totalInputPow   = electronInputSum+ionInputSum
totalPowsrc     = totalInputPow+totalDissNeut
#----- Power Imbalance
powerImbalance = totalPowsrc.values[-1]-totalPowLoss.values[-1]
powerImbPerc   = 100.*(totalPowsrc.values[-1]-totalPowLoss.values[-1])/totalPowsrc.values[-1]

#----- Input Power variation
if(1):
    import os
    tempVal  = 5.0 # eV
    tempMean = (ds["Te"].values+ds["Td+"].values)/2.
    detLoc   = ds["pos"].values[-1]-ds["pos"].values[np.argmin(np.abs(tempMean-tempVal),axis=1)]
    plt.figure()
    plt.title(filePath.split("/")[-2])
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(ds["t"].values-ds["t"].values[0],totalInputPow.values*1e-6,     color="tab:blue",  label="Input Power")
    # ax1.plot(ds["t"].values-ds["t"].values[0],electronInputSum*1e-6,         color="tab:red",   label="Pe_src")
    # ax1.plot(ds["t"].values-ds["t"].values[0],ionInputSum*1e-6,              color="tab:orange",label="Pd+_src")
    ax1.axhline(y=772,color="tab:purple",linestyle="--",label="Base Power (772 MW)")
    # ax1.legend(loc="best")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Input Power (MW)")
    ax1.grid(True)
    # #----- Pow Factor
    # ax2.plot(ds["t"].values-ds["t"].values[0],totalInputPow.values*1e-6/772.,color="white",     alpha=0.0)
    # ax2.set_ylabel("Power Factor")
    #----- Det Loc
    ax2.plot(ds["t"].values-ds["t"].values[0],detLoc, color="tab:red", label="Det Loc")
    ax2.set_ylabel("Detachment Front Location (m)")

    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2,labs1+labs2,loc="best")

    plt.show()

#----- Total energy needed to ionise all the neutrals
# detLoc  = []
# tempVal = 7.0 # eV
# for i in range(len(dataDict["t_array"])):
#     tempHR = np.interp(cellPoshR,cellPos,(dataDict["Te"][i,:]+dataDict["Td+"][i,:])/2.)
#     detPos = cellPoshR[-1]-cellPoshR[np.argmin(np.abs(tempHR-tempVal))]
#     detLoc.append(detPos)
tempMean        = (ds["Te"].values+ds["Td+"].values)/2.
detLoc          = ds["pos"].values[-1]-ds["pos"].values[np.argmin(np.abs(tempMean-5.0),axis=1)]
timeIndex       = 0#np.argmin(detLoc)
E_ion           = 30.*q_e
totalIoniseEng  = (ds["Nd"]*ds["dv"]*E_ion).sum("pos")
powIncFactor    = float(options["Pe"]["powFactor"])
deltaPower      = (powIncFactor-1.)*(electronInputSum+ionInputSum).values[0]
timeToIonise    = totalIoniseEng.values[timeIndex]/deltaPower # J/(J/s) = s
print("Input power = %.3e W, total energy needed to ionise neutrals %.3e J"%(totalPowsrc.values[-1],totalIoniseEng.values[timeIndex]))
# sys.exit()
print("Delta Power = %.3e W (factor %.3fx increase), time needed (for zero additional losses) = %.3e ms"%(deltaPower,powIncFactor,timeToIonise*1e3))
# sys.exit()

print("\nSources")
print("-----")
print("{:<20} {:.3e}".format("Electron Source: ",  electronInputSum.values[-1]))
print("{:<20} {:.3e}".format("Ion Source: ",       ionInputSum.values[-1]))
print("{:<20} {:.3e}".format("Recycled Neutrals: ",G_t.values[-1]))

print("\nLosses")
print("-----")
print("{:<20} {:.3e}".format("Excitation: ",     totalRadHexPowSum.values[-1]))
print("{:<20} {:.3e}".format("Recombination: ",  totalRadHrecPowSum.values[-1]))
print("{:<20} {:.3e}".format("Electorn sheath: ",Q_t_electrons.values[-1]))
print("{:<20} {:.3e}".format("Ion sheath: ",     Q_t_ions.values[-1]))

print("\nPower balance")
print("-----")
print("{:<20} {:.3e}".format("Total Source: ",totalPowsrc.values[-1]))
print("{:<20} {:.3e}".format("Total Losses: ",totalPowLoss.values[-1]))
print("{:<20} {:.3e} / {:.1f}%".format("Power Imbalance: ",powerImbalance,powerImbPerc))

plt.figure()
#----- Sources
plt.plot(ds["t"].values,totalPowsrc.values,       label="Total Source", color="firebrick")
plt.plot(ds["t"].values,totalInputPow.values,     label="Input Power",  color="tab:orange",linestyle="--")
plt.plot(ds["t"].values,totalDissNeut,            label="Nuetral Diss.",color="crimson",   linestyle="--")
#----- Sinks
plt.plot(ds["t"].values,totalPowLoss.values,      label="Total Losses",       color="midnightblue")
plt.plot(ds["t"].values,totalRadArPowSum.values,  label="Ar. Rad.",           color="mediumblue")
plt.plot(ds["t"].values,totalRadHexPowSum.values, label="H.Exc. Rad.",        color="mediumpurple")
plt.plot(ds["t"].values,totalRadHrecPowSum.values,label="H.Rec. Rad",         color="darkorchid")
plt.plot(ds["t"].values,Q_t_ions,                 label="Ion sheath trans.",  color="tab:purple")
plt.plot(ds["t"].values,Q_t_electrons,            label="Elec. sheath trans.",color="royalblue")
# plt.ylim([-0.1e9,1.1e9])
plt.legend(loc="best",ncol=2)
plt.xlabel("Time (ms)")
plt.ylabel("Power (W)")
# plt.show()

powerLossSpace = totalRadArPow+totalRadHexPow+totalRadHrecPow
powerSrceSpace = electronInput+ionInput
powerDiffSpace = powerSrceSpace-powerLossSpace
X, Y = np.meshgrid(ds["pos"].values,ds["t"])
plt.figure()
plt.title("Power Sinks")
plt.pcolormesh(X,Y,powerLossSpace,cmap='viridis')
# plt.colorbar()
plt.xlim(left=0)
plt.xlabel("Cell Pos (m)")
plt.ylabel("Time (ms)")
plt.figure()
plt.title("Power Sources")
plt.pcolormesh(X,Y,powerSrceSpace,cmap='viridis')
plt.xlim(left=0)
plt.xlabel("Cell Pos (m)")
plt.ylabel("Time (ms)")
plt.figure()
plt.title("Net Power")
plt.pcolormesh(X,Y,powerDiffSpace,cmap='viridis')
plt.xlim(left=0)
plt.xlabel("Cell Pos (m)")
plt.ylabel("Time (ms)")
plt.show()
