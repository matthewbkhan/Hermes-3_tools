# from boututils.showdata import showdata
# import xhermes
from boutdata.data import BoutData
import matplotlib.pyplot as plt
from boutdata import collect
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import numpy as np
import sys,os

# This in-place replaces the points in the guard cells with the points on the boundary
def replace_guards(var):
    var[0]  = 0.5*(var[0] + var[1])
    var[-1] = 0.5*(var[-1] + var[-2])
    return var

# Load in the shot data and input options
filePath = sys.argv[1]
fileName = filePath.split("/")[-2]
boutdata = BoutData(filePath, yguards=True, strict = True, DataFileCaching=False)
options  = boutdata["options"]
# ds       = xhermes.open(filePath)

""" Path to the directory containing the data """

dataDict = dict()
dataDict["J"]    = collect("J",path=filePath,yguards=True,info=False,strict=True)[0,1:-1]
dataDict["g_22"] = collect("g_22",path=filePath,yguards=True,info=False,strict=True)[0,1:-1]

# Make the cell position grid
try:
    xPoint     = options["mesh"]["length_xpt"]
except Exception as eErr:
    print("No deinfed x-point")
    xPoint = 35.14
# dymin      = options["mesh"]["dymin"]
# length     = options["mesh"]["length"]
dy         = collect("dy",path=filePath,yguards=True,info=False,strict=True)[0,1:-1]
cellPos    = np.zeros(len(dy))
cellPos[0] = -0.5*dy[1]
cellPos[1] =  0.5*dy[1]
for i in range(2,len(dy)):
    cellPos[i] = cellPos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]
cellEdges = np.zeros(len(dy))
counter   = 0
for i,cellWidth in enumerate(dy[1:-1]):
    counter += cellWidth
    cellEdges[i] = counter

# Normalisation factors
Nnorm    = collect("Nnorm",   path=filePath)
Cs0      = collect("Cs0",     path=filePath)
rho_s0   = collect("rho_s0",  path=filePath)
Omega_ci = collect("Omega_ci",path=filePath)
Tnorm    = collect("Tnorm",   path=filePath)
Pnorm    = Nnorm*Tnorm*1.602e-19 # Converts p to Pascals

if not(isinstance(options["timestep"], int)):
    try:
        options["timestep"] = eval(options["timestep"])
    except eE as Exception:
        print(eE)
        print("Timestep error")

# Cell volume is the volume of each cell, stepTime is the real time of each output step
# J must be normalised by rho_s0^2 as it is effectively an area and x,y=1
cellVolume = dataDict["J"]*dy/dataDict["g_22"]#*rho_s0*rho_s0
stepTime   = options["timestep"]/Omega_ci

pump = False
if(0 and pump):
    dataDict["Sd_src"]  = collect("Sd_src",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Sd+_src"] = collect("Sd+_src",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    #----- Plotting check
    yCode = cellPos*(2.0*np.pi)/100.
    pumpStart = 68 # options["Nd"]["pumpStart"]
    pumpStop  = 73 # options["Nd"]["pumpStop"]
    y_pumpStart = np.pi * ( 2. - dymin - np.sqrt( (2.-dymin)**2. - 4.*(1.-dymin)*pumpStart/length ) ) / (1.0 - dymin)
    y_pumpStop  = np.pi * ( 2. - dymin - np.sqrt( (2.-dymin)**2. - 4.*(1.-dymin)*pumpStop/length ) ) / (1.0 - dymin)
    #----- Pump Volumes
    def tanh(x,x0):
        return (np.exp(2.*(x-x0))-1.)/(np.exp(2.*(x-x0))+1.)
    sourceVolume   = np.sum(cellVolume[1:-1][:np.argmin(np.abs(cellPos-xPoint))])
    pumpRegion  = (0.5*tanh(cellPos,pumpStart)+0.5)*(-0.5*tanh(cellPos,pumpStop)+0.5)
    pumpVolume  = np.sum(pumpRegion*cellVolume)#/np.max(pumpRegion))
    print(pumpVolume)
    print(sourceVolume)
    print(sourceVolume/pumpVolume)
    print(np.max(pumpRegion))
    # pumpVolumeTanh = np.sum(cellVolume[1:-1]*pumpRegion)
    plt.figure()
    plt.plot(cellPos,np.abs(dataDict["Sd_src"][-1,:]))
    # plt.plot(cellPos,pumpRegion)#(pumpRegion/np.max(pumpRegion))*np.max(np.abs(dataDict["Sd_src"][-1,:])))
    plt.figure()
    plt.imshow(np.abs(collect("Sd_src",path=filePath,yguards=True,info=False,strict=True)[:,0,:,0]))
    plt.figure()
    plt.imshow(np.abs(collect("Sd+_src",path=filePath,yguards=True,info=False,strict=True)[:,0,:,0]))
    plt.show()

# Timing data
dataDict["t_array"]    = collect("t_array", path=filePath,yguards=True,info=False,strict=True)
dataDict["iteration"]  = collect("t_array", path=filePath,yguards=True,info=False,strict=True)
dataDict["wtime"]      = collect("wtime",   path=filePath,yguards=True,info=False,strict=True)
dataDict["ncalls"]     = collect("ncalls",  path=filePath,yguards=True,info=False,strict=True)

# Normalise to time array
tind = np.copy(dataDict["t_array"]/options["timestep"])
dataDict["t_array"] /= Omega_ci # stepTime/options["timestep"]
dataDict["t_array"] -= dataDict["t_array"][0]
dataDict["t_array"] *= 1e3
# dataDict["t_array"] -= 500
# print(Omega_ci)
# print(dataDict["t_array"][1])
# # print(dataDict["t_array"][1:]-dataDict["t_array"][:-1])
# print(options["timestep"])
# sys.exit()

""" Load all variables """
dataDict["Nd"]      = collect("Nd",    path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Nd+"]     = collect("Nd+",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Ne"]      = collect("Ne",    path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

dataDict["Pd"]       = collect("Pd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Pd+"]      = collect("Pd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Pe"]       = collect("Pe",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Pe_src"]   = collect("Pe_src",path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

dataDict["Td"]       = collect("Td",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Td+"]      = collect("Td+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Te"]       = collect("Te",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
#
# dataDict["NVd"]       = collect("NVd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["NVd+"]      = collect("NVd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
""" NVe == NVd+ """
dataDict["NVe"]       = collect("NVd+",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

# Shape is [0,time,0]???
# dataDict["Vd"]       = collect("Vd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Vd+"]      = collect("Vd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
dataDict["Ve"]       = collect("Ve",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

stillFrames = False

atomics = True
if(atomics):
    dataDict["Rar"]       = collect("Rar",     path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Rd+_ex"]    = collect("Rd+_ex",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Rd+_rec"]   = collect("Rd+_rec", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    dataDict["Ed+_iz"]    = collect("Ed+_iz",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Ed+_rec"]   = collect("Ed+_rec", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Ed_Dpar"]   = collect("Ed_Dpar", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Edd+_cx"]   = collect("Edd+_cx", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    dataDict["Fd+_iz"]    = collect("Fd+_iz",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Fd+_rec"]   = collect("Fd+_rec", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Fd_Dpar"]   = collect("Fd_Dpar", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Fdd+_cx"]   = collect("Fdd+_cx", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    dataDict["Sd+_iz"]    = collect("Sd+_iz",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Sd+_rec"]   = collect("Sd+_rec", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Sd+_src"]   = collect("Sd+_src", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Sd_Dpar"]   = collect("Sd_Dpar", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

ddt = False
if(ddt):
    dataDict["ddt(NVd+)"] = collect("ddt(NVd+)",path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["ddt(Nd)"]   = collect("ddt(Nd)",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["ddt(Nd+)"]  = collect("ddt(Nd+)", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["ddt(Pd)"]   = collect("ddt(Pd)",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["ddt(Pd+)"]  = collect("ddt(Pd+)", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["ddt(Pe)"]   = collect("ddt(Pe)",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

# Apply normalisations
# dataDict["Te"]  = (0.5*dataDict["P"]/dataDict["Ne"]) * Tnorm
dataDict["Pd"]   *= Pnorm
dataDict["Pd+"]  *= Pnorm
dataDict["Pe"]   *= Pnorm

dataDict["Nd"]   *= Nnorm
dataDict["Nd+"]  *= Nnorm
dataDict["Ne"]   *= Nnorm

dataDict["Td"]   *= Tnorm
dataDict["Td+"]  *= Tnorm
dataDict["Te"]   *= Tnorm

# dataDict["NVd"]  *= Cs0*Nnorm
dataDict["NVd+"] *= Cs0*Nnorm
dataDict["NVe"]  *= Cs0*Nnorm

# dataDict["Vd"]   *= dataDict["NVd"]/dataDict["Nd"]
dataDict["Vd+"]  *= dataDict["NVd+"]/dataDict["Nd+"]
dataDict["Ve"]   *= dataDict["NVe"]/dataDict["Ne"]

# Replace guards for relevant data
for tind in range(len(dataDict["t_array"])):
    dataDict["Nd"][tind,:]   = replace_guards(dataDict["Nd"][tind,:])
    dataDict["Nd+"][tind,:]  = replace_guards(dataDict["Nd+"][tind,:])
    dataDict["Ne"][tind,:]   = replace_guards(dataDict["Ne"][tind,:])
    """ Don't guard replace NVi, see the original profiles.py file """
    # dataDict["NVi"][tind,:] = replace_guards(dataDict["NVi"][tind,:])

    # dataDict["Vd"][tind,:]  = replace_guards(dataDict["Vd"][tind,:])
    # dataDict["Vd+"][tind,:] = replace_guards(dataDict["Vd+"][tind,:])
    # dataDict["Ve"][tind,:]  = replace_guards(dataDict["Ve"][tind,:])

    dataDict["Pd"][tind,:]  = replace_guards(dataDict["Pd"][tind,:])
    dataDict["Pd+"][tind,:] = replace_guards(dataDict["Pd+"][tind,:])
    dataDict["Pe"][tind,:]  = replace_guards(dataDict["Pe"][tind,:])

    if(atomics):
        dataDict["Rar"][tind,:]     = replace_guards(dataDict["Rar"][tind,:])
        dataDict["Rd+_ex"][tind,:]  = replace_guards(dataDict["Rd+_ex"][tind,:])
        dataDict["Rd+_rec"][tind,:] = replace_guards(dataDict["Rd+_rec"][tind,:])

        dataDict["Ed+_iz"][tind,:]  = replace_guards(dataDict["Ed+_iz"][tind,:])
        dataDict["Ed+_rec"][tind,:] = replace_guards(dataDict["Ed+_rec"][tind,:])
        dataDict["Ed_Dpar"][tind,:] = replace_guards(dataDict["Ed_Dpar"][tind,:])
        dataDict["Edd+_cx"][tind,:] = replace_guards(dataDict["Edd+_cx"][tind,:])

        dataDict["Fd+_iz"][tind,:]  = replace_guards(dataDict["Fd+_iz"][tind,:])
        dataDict["Fd+_rec"][tind,:] = replace_guards(dataDict["Fd+_rec"][tind,:])
        dataDict["Fd_Dpar"][tind,:] = replace_guards(dataDict["Fd_Dpar"][tind,:])
        dataDict["Fdd+_cx"][tind,:] = replace_guards(dataDict["Fdd+_cx"][tind,:])

        dataDict["Sd+_iz"][tind,:]  = replace_guards(dataDict["Sd+_iz"][tind,:])
        dataDict["Sd+_rec"][tind,:] = replace_guards(dataDict["Sd+_rec"][tind,:])
        dataDict["Sd_Dpar"][tind,:] = replace_guards(dataDict["Sd_Dpar"][tind,:])
        dataDict["Sd+_src"][tind,:] = replace_guards(dataDict["Sd+_src"][tind,:])

    dataDict["Td"][tind,:]   = replace_guards(dataDict["Td"][tind,:])
    dataDict["Td+"][tind,:]  = replace_guards(dataDict["Td+"][tind,:])
    dataDict["Te"][tind,:]   = replace_guards(dataDict["Te"][tind,:])
    # dataDict["Td"][tind,-1]  = dataDict["Td"][tind,-2] # Zero-gradient Te, from original profiles.py
    # dataDict["Td+"][tind,-1] = dataDict["Td+"][tind,-2] # Zero-gradient Te, from original profiles.py
    # dataDict["Te"][tind,-1]  = dataDict["Te"][tind,-2] # Zero-gradient Te, from original profiles.py

# Line averaged Denisty
if(1):
    totalElectronNumber = np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalIonNumber      = np.sum(dataDict["Nd+"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalNeutralNumber  = np.sum(dataDict["Nd"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalParticleNumber = totalIonNumber+totalNeutralNumber
    meanParticleNumber  = totalParticleNumber/np.sum(cellVolume[1:-1])
    meanNeutralNumber   = totalNeutralNumber/np.sum(cellVolume[1:-1])
    meanIonNumber       = totalIonNumber/np.sum(cellVolume[1:-1])
    timeIndex = -1
    print("Mean total Particle number (d,d+): initial = %.3e, at time index %i = %.3e"%(meanParticleNumber[timeIndex],timeIndex,meanParticleNumber[-1]))
    print("Mean total Ion number:             initial = %.3e, at time index %i = %.3e"%(meanIonNumber[timeIndex],     timeIndex,meanIonNumber[-1]))
    print("Mean total Neutral number:         initial = %.3e, at time index %i = %.3e"%(meanNeutralNumber[timeIndex], timeIndex,meanNeutralNumber[-1]))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax2.plot(meanParticleNumber,color="white",alpha=0)
    ax1.plot(dataDict["t_array"],meanParticleNumber,label="Total mean")
    ax1.plot(dataDict["t_array"],meanNeutralNumber, label="Neutral mean")
    ax1.plot(dataDict["t_array"],meanIonNumber,     label="Ion mean")
    ax1.set_xlabel("Sim. Time (ms)")
    ax2.set_xlabel("Sim. tind")
    ax1.set_ylabel("Mean Particle Number")
    ax1.legend(loc="best")
    plt.tight_layout()
    plt.show()

#----- Failures
if(0):
    scale = 1.2
    figsize = (8*scale,4*scale)
    dpi = 150/scale
    fig, axes = plt.subplots(2,4, figsize=figsize, dpi = dpi)

    fig.subplots_adjust(hspace=0.4, top = 0.85)
    fig.suptitle(casename)

    lw = 2
    axes[1,0].plot(t, wtime_per_stime/1e4, c = "k", lw = lw)
    axes[1,0].set_title("wtime/(1e4*stime)")
    # axes[1,0].set_yscale("log")
    axes[1,1].plot(t, lratio, c = "k", lw = lw)
    axes[1,1].set_title("linear/nonlinear")
    axes[1,2].plot(t, np.clip(fails, 0, np.max(fails)), c = "k", lw = lw)
    axes[1,2].set_title("nfails")
    axes[1,2].set_ylim(0,None)
    # axes[1,2].set_yscale("log")
    axes[1,3].plot(t, lorder, c = "k", lw = lw)
    axes[1,3].set_title("order")

#----- Ionisation and Recombination
if(0):
    """ ############## """
    """ NEEDS EDITTING """
    """ ############## """
    import sys
    sys.path.insert(1, '/home/mbk513/Documents/Projects/STEP_lot2/AMJUEL') # caution: path[0] is reserved for script path (or '' in REPL)
    import AMJUEL_loadPlot as amjuel

    Te = np.logspace(0,4,num=1000)
    Ne = 3e20
    ionRate = amjuel.amjuel_2d("/home/mbk513/Documents/Projects/STEP_lot2/AMJUEL/AMJUEL_ionisationRate.csv",Te,Ne,mode="default",second_param="Ne",densUnits="m")
    tIndex = 1
    dataDict["Sd+_iz"]  *= Nnorm*Omega_ci
    dataDict["Sd+_rec"] *= Nnorm*Omega_ci

    plt.figure()
    tempPeakIz = dataDict["Te"][np.arange(np.shape(dataDict["Te"])[0]),np.argmax(dataDict["Sd+_iz"],axis=1)]
    tempMeanIz = np.sum(dataDict["Sd+_iz"]*dataDict["Te"],axis=1)/np.sum(dataDict["Sd+_iz"],axis=1)
    densMeanIz = np.sum(dataDict["Sd+_iz"]*dataDict["Ne"],axis=1)/np.sum(dataDict["Sd+_iz"],axis=1)
    plt.plot(dataDict["t_array"],tempMeanIz)
    plt.plot(dataDict["t_array"],densMeanIz)
    plt.figure()
    S_cd = [amjuel.amjuel_2d("/home/mbk513/Documents/Projects/STEP_lot2/AMJUEL/AMJUEL_ionisationRate.csv",Te,Ne) for Te,Ne in zip(tempMeanIz,densMeanIz)]
    plt.plot(dataDict["t_array"],densMeanIz*S_cd)
    plt.figure()
    plt.plot(dataDict["t_array"],np.sqrt((tempMeanIz*1.6e-19)/(2*1.67e-27)))
    plt.show()

    # densNormed = np.max(dataDict["Sd+_iz"][tIndex,:])*dataDict["Ne"][tIndex,:]/np.max(dataDict["Ne"][tIndex,:])
    # tempNormed = np.max(dataDict["Sd+_iz"][tIndex,:])*dataDict["Te"][tIndex,:]/np.max(dataDict["Te"][tIndex,:])
    # print(dataDict["Te"][tIndex,np.argmax(dataDict["Sd+_iz"][tIndex,:])])
    # print(dataDict["Ne"][tIndex,np.argmax(dataDict["Sd+_iz"][tIndex,:])])
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # ax1.plot(cellPos, dataDict["Sd+_iz"][tIndex,:], color="tab:red",   linestyle="-", label=" Sd+_iz")
    # ax1.plot(cellPos,-dataDict["Sd+_rec"][tIndex,:],color="tab:orange",linestyle="--",label="-Sd+_rec")
    # ax1.plot(cellPos, densNormed,                   color="tab:blue", label="normed Ne",linestyle=":")
    # ax1.plot(cellPos, tempNormed,                   color="tab:green",label="normed Te",linestyle=":")
    # ax1.set_ylabel("Plasma Source/Sinks (m$^{-3}$s$^{-1}$)")
    # # ax2.plot(cellPos,dataDict["Ne"][1,:],color="tab:blue",label="Ne",linestyle=":")
    # ax2.set_ylabel("Plasma Density")
    # ax1.set_xlabel("Cell Pos (m)")
    # ax1.legend(loc="best")
    # ax1.set_zorder(2)
    # ax2.set_zorder(1)
    # ax1.patch.set_visible(False)
    # plt.show()
    # # sys.exit()

# ddt comparison
if(ddt):
    params  = ["ddt(NVd+)","ddt(Nd)","ddt(Nd+)","ddt(Pd)","ddt(Pd+)","ddt(Pe)"]
    colours = cm.rainbow(np.linspace(0,1,len(params)+1))
    mpl.rcParams['font.size']=14
    plt.figure()
    for param,colour in zip(params,colours):
        cw_ddt  = dataDict[param]*cellVolume/np.sum(cellVolume)
        rms_ddt = np.sqrt(np.mean(cw_ddt**2, axis = 1))
        plt.plot(dataDict["t_array"][1:],rms_ddt[1:],color=colour,label=param)
    plt.yscale("log")
    plt.legend(loc="best")
    plt.ylabel("rms(ddt)")
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

# Heat conduction stuff
if(0):
    def heatConduction(pos,T,kappa0,tIndex):
        grad_T   = (T[tIndex,1:] - T[tIndex,:-1]) / (pos[1:] - pos[:-1]) # One-sided differencing
        T_p      = 0.5*(T[tIndex,1:] + T[tIndex,:-1])
        kappa0_p = kappa0 # 0.5*(kappa0[1:] + kappa0[:-1])
        # Position of the result
        result_pos = 0.5*(pos[1:] + pos[:-1])
        q_par      = -kappa0_p*T_p**(5./2)*grad_T
        return result_pos, T_p, grad_T, q_par

    colours = ["tab:red","tab:green","tab:blue","black"]
    for tIndex,colour in zip([0,1,2,-1],colours):
        result_pos, T_p, grad_T, q_par = heatConduction(cellPos,dataDict["Td+"],2000.,tIndex)
        plt.plot(result_pos,-grad_T,color=colour, linestyle="-", label="-grad_T")
        plt.plot(result_pos,T_p,    color=colour, linestyle="--",label="T_p")
        plt.plot(result_pos,q_par,  color=colour, linestyle=":", label="q_par")
    # plt.legend(loc="best")
    plt.show()
    sys.exit()

# wtime check
if(0):
    print(np.sum(dataDict["wtime"]))
    print(np.sum(dataDict["wtime"][:300]))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    dataDict["cvode_last_order"]   = collect("cvode_last_order", path=filePath,yguards=True,info=False,strict=True)
    ax2.plot(dataDict["t_array"], dataDict["cvode_last_order"],color="tab:blue")
    ax1.plot(dataDict["t_array"], dataDict["wtime"],           color="tab:red")#,label=fileName.split("/")[2])
    print("Run time: ")
    print("Seconds: %i"%(np.sum(dataDict["wtime"])))
    print("Minutes: %i"%(np.sum(dataDict["wtime"])/60.))
    print("Hours:   %i"%(np.sum(dataDict["wtime"])/3600.))
    ax1.set_zorder(1)
    ax2.set_zorder(-1)
    # ax1.legend(loc="best")
    # ax1.set_yscale("log")
    ax1.set_xlabel("Sim. Time (ms)")
    ax2.set_xlabel("Sim. tind")
    ax1.set_ylabel("Run Time (s)")
    ax1.patch.set_visible(False)
    plt.tight_layout()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax2.plot(dataDict["ncalls"],color="white",alpha=0)
    ax1.plot(dataDict["t_array"], dataDict["ncalls"],color="tab:blue")
    ax1.set_zorder(1)
    ax2.set_zorder(-1)
    # ax1.legend(loc="best")
    ax1.set_xlabel("Sim. Time (ms)")
    ax2.set_xlabel("Sim. tind")
    ax1.set_ylabel("ncalls")
    plt.tight_layout()
    # plt.show()

    if(0):
        dataDict["wall_time"]          = collect("wall_time", path=filePath,yguards=True,info=False,strict=True)
        dataDict["wtime_per_rhs"]      = collect("wtime_per_rhs", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_nsteps"]       = collect("cvode_nsteps", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_nfevals"]      = collect("cvode_nfevals", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_nniters"]      = collect("cvode_nniters", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_nliters"]      = collect("cvode_nliters", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_last_order"]   = collect("cvode_last_order", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_num_fails"]    = collect("cvode_num_fails", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_nonlin_fails"] = collect("cvode_nonlin_fails", path=filePath,yguards=True,info=False,strict=True)
        dataDict["cvode_last_step"]    = collect("cvode_last_step", path=filePath,yguards=True,info=False,strict=True)

        dataDict["cvode_num_fails"][1:] = [dataDict["cvode_num_fails"][i]-dataDict["cvode_num_fails"][i-1] for i in range(1,len(dataDict["cvode_nsteps"]))]

        params = ["wtime","wall_time","ncalls","wtime_per_rhs","cvode_nsteps",
                  "cvode_nfevals","cvode_nniters","cvode_last_order","cvode_num_fails",
                  "cvode_nonlin_fails","cvode_last_step"]
        for param,colour in zip(params,cm.rainbow(np.linspace(0,1,len(params)+1))):
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            ax2.plot(dataDict[param],color="white",alpha=0)
            ax1.plot(dataDict["t_array"]*1e3, dataDict[param],color=colour,label=param)#,label=fileName.split("/")[2])
            plt.title(param)
            ax1.set_xlabel("Sim. Time (ms)")
            ax2.set_xlabel("Sim. tind")
            ax1.set_ylabel(param)
            plt.tight_layout()
            plt.show()

# Det Front position
if(0):
    cellPoshR = np.arange(cellPos[0],cellPos[-1],0.001)
    detLoc = []
    for i in range(len(dataDict["t_array"])):
        tempHR = np.interp(cellPoshR,cellPos,dataDict["Te"][i,:])
        detPos = cellPoshR[-1]-cellPoshR[np.argmin(np.abs(tempHR-5.0))]
        detLoc.append(detPos)

    totalElectronNumber = np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalIonNumber      = np.sum(dataDict["Nd+"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalNeutralNumber  = np.sum(dataDict["Nd"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalParticleNumber = totalNeutralNumber+totalIonNumber
    meanParticleNumber  = totalParticleNumber/np.sum(cellVolume[1:-1])
    for detPoscm in [495,995,1495]:
        detArg     = np.argmin(np.abs(1e2*np.array(detLoc)-detPoscm))
        meanDens   = meanParticleNumber[detArg]
        plas, neut = totalIonNumber[detArg]/np.sum(cellVolume[1:-1]), totalNeutralNumber[detArg]/np.sum(cellVolume[1:-1])
        print("Det. Front Position: %.1f, at time index: %i, mean density = %.3e, ion dens = %.3e, neutral dens = %.3e"%(detPoscm,detArg,meanDens,plas,neut))

# Timescales and speeds
if(0):
    tInd        = 0
    conLen      = options["mesh"]["length"]
    gamma       = 1.67
    soundSpeed  = np.sqrt(gamma*1.*1.6e-19*dataDict["Td+"][tInd,:]/(2.*1.67e-27))
    eleThermVel = np.sqrt(1.6e-19*dataDict["Te"][tInd,:]/9.11e-31)
    ionThermVel = np.sqrt(1.6e-19*dataDict["Td+"][tInd,:]/(2.*1.67e-27)) # N.B. This is a model of a 1D fluid, assuming equal ion and electron temperatures
    print(r"sound speed      : mean = %.3e m/s, frac. int. = %.3e m/s, tau_conv ~ L/C_s  ~ %.1f ms / %.1f us"%(np.mean(soundSpeed), np.trapz(soundSpeed,cellPos[:]/conLen), 1e3*conLen/np.trapz(soundSpeed,cellPos[:]/conLen), 1e6*conLen/np.trapz(soundSpeed,cellPos[:]/conLen)))
    print(r"electron thermal : mean = %.3e m/s, frac. int. = %.3e m/s, tau_cond ~ L/v_eT ~ %.1f ms / %.1f us"%(np.mean(eleThermVel),np.trapz(eleThermVel,cellPos[:]/conLen),1e3*conLen/np.trapz(eleThermVel,cellPos[:]/conLen),1e6*conLen/np.trapz(eleThermVel,cellPos[:]/conLen)))
    print(r"ion thermal      : mean = %.3e m/s, frac. int. = %.3e m/s, tau_cond ~ L/v_iT ~ %.1f ms / %.1f us"%(np.mean(ionThermVel),np.trapz(ionThermVel,cellPos[:]/conLen),1e3*conLen/np.trapz(ionThermVel,cellPos[:]/conLen),1e6*conLen/np.trapz(ionThermVel,cellPos[:]/conLen)))
    print("cond. time =  %.1f us"%(1e6*np.sum(dy/eleThermVel)))
    print("conv. time = %.1f us"%(1e6*np.sum(dy/np.mean(np.sqrt(gamma*1.*1.6e-19*dataDict["Td+"][tInd,:]/(2.5*1.67e-27)),axis=0))))
    plt.plot(soundSpeed,label="sound speed")
    plt.plot(eleThermVel,label="electron thermal")
    plt.plot(ionThermVel,label="ion thermal")
    plt.legend(loc="best")
    # plt.show()

# Plasma momentum plot
if(0):
    numSteps = 6
    deutMass = 2.014*1.67262192e-27
    profiles = [deutMass*dataDict["NVi"],dataDict["Fcx"],-dataDict["Siz"],dataDict["Srec"]]
    yLabels  = ["Ion Momentum (kg m s^-1)","Fcx","-Siz","Srec"]
    titles   = ["Ion Mom.",                "Fcx","-Siz","Srec"]
    for figNum,profile,yLabel,title in zip(range(len(profiles)),profiles,yLabels,titles):
        fig = plt.figure(figNum,figsize=(7,5))
        mpl.rcParams['font.size']=14
        plt.title(title)
        for i,colour in zip(range(-numSteps,0,1),cm.viridis(np.linspace(0,1,numSteps+1))[::-1]):
            plt.plot(cellPos,profile[i,:],color=colour,linewidth=1,linestyle="-",label="tind=%i"%i)
        plt.xlabel("cellPos (m)")
        plt.ylabel(yLabel)
        plt.xlim([77.44,78.14])
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
    plt.show()
    # sys.exit()

# Heat flux on target
if(0):
    bohmVel = np.sqrt(2.*1.6e-19*dataDict["Te"]/(2.5*1.67e-27))
    maxVel  = [np.max([i,j]) for i,j in zip(dataDict["Vd+"][:,-1],bohmVel[:,-1])]
    dataDict["Q_t"] = 7.0*dataDict["Ne"][:,-1]*(1.6e-19)*dataDict["Te"][:,-1]*maxVel

    fig = plt.figure(figsize=(9,6))
    mpl.rcParams['font.size']=14
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()
    t0  = 0.#5.
    ax2.plot(np.arange(len(dataDict["Q_t"]))-0.,1e-6*dataDict["Q_t"],     color="white",   linewidth=2,linestyle="-", alpha=0)
    ax1.plot((dataDict["t_array"]*1e3)-t0,      1e-6*dataDict["Q_t"],     color="tab:blue",linewidth=2,linestyle="-", label="Target Heat Flux",zorder=10)#,marker="x")
    # ax1.plot(np.array(inputTimes)-0.,             1e-6*np.array(inputPower),color="k",       linewidth=2,linestyle="--",label="Input Power")
    ax1.plot((dataDict["t_array"]*1e3)-t0,      1e+0*dataDict["Te"][:,-1],color="tab:red", linewidth=2,linestyle=":", label="Target Temp. (eV)")
    # plt.xlim([dataDict["t_array"][300]*1e3,dataDict["t_array"][508]*1e3])
    # ax2.set_xlim([-250,750])
    # ax1.set_xlim([0,5])
    # ax1.set_ylim([1e-2,4e0])
    # plt.ylim([6.5e5,1.15e10])
    ax1.set_yscale("log")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Target Heat Flux (MW/m$^2$)")
    ax3.set_ylabel("Transient Power (GW)")
    ax1.legend(loc="upper left")
    ax2.set_xlabel("Time index")
    # ax1.set_zorder(1)
    # ax2.set_zorder(-1)

    # fig = plt.figure(figsize=(9,6))
    # mpl.rcParams['font.size']=14
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    # ax2.plot(dataDict["NVi"][:,-1]/1e23,color="white",linestyle=" ",alpha=0)
    # ax1.plot(dataDict["t_array"]*1e3,dataDict["NVi"][:,-1]/1e23,label="NVi[targ] (1e23 ms-2 / m^-3)")#/np.max(dataDict["NVi"][:,-1]),label="NVi")
    # # ax1.plot(dataDict["t_array"]*1e3,dataDict["Vi"][:,-1],  label="Vi")#/np.max(dataDict["Vi"][:,-1]),  label="Vi")
    # ax1.plot(dataDict["t_array"]*1e3,dataDict["Ne"][:,-1]/1e20,  label="Ne[targ] (1e20 m^-3)")#/np.max(dataDict["Ne"][:,-1]),  label="Ne")
    # ax1.plot(dataDict["t_array"]*1e3,dataDict["Te"][:,-1],  label="Te[targ] (eV)")#/np.max(dataDict["Te"][:,-1]),  label="Te")
    # ax1.set_xlabel("Time (ms)")
    # ax1.set_xlabel("tind")
    # ax1.set_ylabel("Normalised Values")
    # ax1.legend(loc="best")
    # plt.tight_layout()
    # plt.show()
    # # sys.exit()

# Radiation checking
if(0):
    tind = -1#354
    plt.figure(4)
    plt.title("Final Density Profiles")
    # Momentum
    plt.plot(cellPos[:],1e-23*dataDict["NVi"][tind,:],linestyle="-", marker="",color="tab:red",  label="NVi",zorder=2)
    plt.plot(cellPos[:],1e-23*dataDict["NVn"][tind,:],linestyle="-", marker="",color="tab:brown",label="NVn",zorder=2)
    # Radiation
    plt.plot(cellPos[:],1e-3*-dataDict["Fcx"][tind,:],linestyle="--",marker="",color="tab:orange",label="Fcx", zorder=0)
    plt.plot(cellPos[:],1e-24*dataDict["Siz"][tind,:],linestyle="--",marker="",color="tab:blue",  label="Siz", zorder=1)
    plt.plot(cellPos[:],6e-25*-dataDict["Srec"][tind,:],linestyle="--",marker="",color="tab:green", label="Srec",zorder=1)
    plt.legend(loc="upper left")
    plt.xlim([77.8,78.15])
    plt.xlabel("cellPos (m)")
    plt.yscale("symlog")
    plt.grid(True)
    # plt.show()
    # sys.exit()

plt.show()

# animation taken from here: https://www.geeksforgeeks.org/create-an-animated-gif-using-python-matplotlib/
if(0):
    import matplotlib.animation as animation
    from IPython import display

    def plotStillFrame(paramsPlot,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels):
        plt.figure()
        for param,colour,mark,lab,linestyle in zip(paramsPlot,colours,markers,labels,linestyles):
            plt.plot(cellPos,param[stillFrame,:],color=colour,marker=mark,label=lab,linestyle=linestyle)
        plt.axvline(x=xPoint,linestyle="--",color="tab:grey")
        plt.title("tind = %i"%(stillFrame))
        plt.xlabel("cellPos (m)")
        plt.ylabel(yLabel)
        plt.legend(loc="best")
        plt.grid(True)
        if(logPlot):plt.yscale("log")
        plt.tight_layout()

    def gifParam(paramsPlot,cellPos,startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels,symLog=False):
        animFig = plt.figure()
        ax1 = plt.gca()
        mpl.rcParams['font.size']=14
        pltLines = []
        for paramNum,colour,mark,lab,linestyle in zip(range(np.shape(paramsPlot)[0]),colours,markers,labels,linestyles):
            pltLines.append(plt.plot([],color=colour,marker=mark,label=lab,linestyle=linestyle)[0])
        # Plot limits
        if not(logPlot):
            yMax, yMin = np.max(paramsPlot), np.min(paramsPlot)
        if(logPlot):
            yMax, yMin = 1.1*np.max(paramsPlot), 0.9*np.min(np.array(paramsPlot)[np.array(paramsPlot)>0.0])
        if(symLog):
            yMax, yMin = 1.1*np.max(paramsPlot), -1.1*np.max(np.abs(paramsPlot))
        # Figure Stuff
        plt.xlim(min(cellPos),1.01*max(cellPos))
        plt.ylim(yMin,yMax)
        animFig.set_tight_layout(True)

        # plt.xlim([77.2,78.2])
        # plt.ylim([0.0,50])
        plt.xlabel("cellPos (m)")
        plt.ylabel(yLabel)
        if(legend):plt.legend(loc="best")
        plt.grid(True)
        if(logPlot):plt.yscale("log")
        if(symLog):plt.yscale("symlog")
        # Pause stuff
        Nframes = np.shape(paramsPlot)[1]
        from matplotlib.backend_bases import MouseButton
        def onPress(event):
            global pause
            pause ^= True
        def onClick(event):
            global frame
            if event.button is MouseButton.LEFT:
                frame -= 1
            if event.button is MouseButton.RIGHT:
                frame += 1
        def control():
            global frame, pause
            if frame == Nframes - 1: frame = -1
            if not pause: frame = frame + 1
            return frame
        def AnimationFunctionDens(i):
            frame = control()
            plt.title("tind = %i"%(startFrame+frame))
            # plt.title("time = %.3f ms"%(dataDict["t_array"][startFrame+frame]))
            for paramNum in range(np.shape(paramsPlot)[0]):
                pltLines[paramNum].set_data((cellPos, paramsPlot[paramNum][frame,:]))
        animFig.canvas.mpl_connect("key_press_event", onPress)
        animFig.canvas.mpl_connect("button_press_event", onClick)
        anim_created = animation.FuncAnimation(animFig, AnimationFunctionDens, frames=Nframes, interval=50)
        return anim_created

    if(atomics):
        """ Radiation """
        params     = [dataDict["Rar"],-dataDict["Rd+_ex"],dataDict["Rd+_rec"]]
        colours    = ["tab:red","tab:green","tab:blue"]
        markers    = ["","",""]
        linestyles = ["-","--",":"]
        labels     = ["Rar","Rd+_ex","Rd+_rec"] # Argon radiation, ion excitation radiation, ion recombination radiation
        legend     = True
        logPlot    = True
        yLabel     = "Radiation Losses (arb units)"
        startFrame = 50
        stopFrame  = np.shape(dataDict["Rar"])[0]
        frame      = -2
        pause      = False
        startXPos  = 0 # 589
        stopXPos   = np.shape(dataDict["Rar"])[1] #801
        paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in params]
        paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
        # plt.tight_layout()
        # f = r"./plasmaRad.gif"
        # writergif = animation.PillowWriter(fps=10)
        # paramsGif.save(f, writer=writergif)
        plt.show()
        if(stillFrames):
            stillFrame = np.shape(dataDict["Rar"])[0]-6
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_radiationLosses_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            stillFrame = np.shape(dataDict["Rar"])[0]-1
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_radiationLosses_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            plt.show()

        """ Sink of Energy """
        params     = [dataDict["Ed+_iz"],-dataDict["Ed+_rec"],dataDict["Ed_Dpar"],dataDict["Edd+_cx"]]
        colours    = ["tab:red","tab:orange","tab:green","tab:blue"]
        markers    = ["","","",""]
        linestyles = ["-","--",":","-."]
        labels     = ["Ed+_iz","Ed+_rec","Ed_Dpar","Edd+_cx"] # Sink of plasma energy due to ionisation, recombintaion, parallel diffusion, charge exchange
        legend     = True
        logPlot    = True
        yLabel     = "Sink of Energy (arb units)"
        startFrame = 50
        stopFrame  = np.shape(dataDict["Ed+_iz"])[0]
        frame      = -2
        pause      = False
        startXPos  = 0
        stopXPos   = np.shape(dataDict["Ed+_iz"])[1]
        paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in params]
        paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
        # f = r"./energySink.gif"
        # writergif = animation.PillowWriter(fps=10)
        # paramsGif.save(f, writer=writergif)
        plt.show()
        if(stillFrames):
            stillFrame = np.shape(dataDict["Ed+_iz"])[0]-6
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_sinkOfEnergy_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            stillFrame = np.shape(dataDict["Ed+_iz"])[0]-1
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_sinkOfEnergy_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            plt.show()

        """ Sink of Momentum """
        params     = [dataDict["Fd+_iz"],-dataDict["Fd+_rec"],dataDict["Fd_Dpar"],dataDict["Fdd+_cx"]]
        colours    = ["tab:red","tab:orange","tab:green","tab:blue"]
        markers    = ["","","",""]
        linestyles = ["-","--",":","-."]
        labels     = ["Fd+_iz","Fd+_rec","Fd_Dpar","Fdd+_cx"] # Sink of plasma momentum due to ionisation, recombintaion, parallel diffusion, charge exchange
        legend     = True
        logPlot    = True
        yLabel     = "Sink of plasma momentum (arb units)"
        startFrame = 50
        stopFrame  = np.shape(dataDict["Fd+_iz"])[0]
        frame      = -2
        pause      = False
        startXPos  = 0
        stopXPos   = np.shape(dataDict["Fd+_iz"])[1]
        paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in params]
        paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
        # f = r"./plasmaMomSink.gif"
        # writergif = animation.PillowWriter(fps=10)
        # paramsGif.save(f, writer=writergif)
        plt.show()
        if(stillFrames):
            stillFrame = np.shape(dataDict["Fd+_iz"])[0]-6
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_sinkOfMomentum_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            stillFrame = np.shape(dataDict["Fd+_iz"])[0]-1
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_sinkOfMomentum_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            plt.show()

        """ Sink of Particles """
        params     = [dataDict["Sd+_iz"],-dataDict["Sd+_rec"],dataDict["Sd_Dpar"],dataDict["Sd+_src"]]
        colours    = ["tab:red","tab:orange","tab:green","tab:blue"]
        markers    = ["","","",""]
        linestyles = ["-","--",":","-."]
        labels     = ["Sd+_iz","Sd+_rec","Sd_Dpar","Sd+_src"] # Sink of plasma particles due to ionisation, recombintaion, parallel diffusion, charge exchange
        legend     = True
        logPlot    = True
        yLabel     = "Sink of particles (arb units)"
        startFrame = 50
        stopFrame  = np.shape(dataDict["Sd+_iz"])[0]
        frame      = -2
        pause      = False
        startXPos  = 0
        stopXPos   = np.shape(dataDict["Sd+_iz"])[1]
        paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in params]
        paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
        # f = r"./plasmaPartsSink.gif"
        # writergif = animation.PillowWriter(fps=10)
        # paramsGif.save(f, writer=writergif)
        plt.show()
        if(stillFrames):
            stillFrame = np.shape(dataDict["Sd+_iz"])[0]-6
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_particleSink_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            stillFrame = np.shape(dataDict["Sd+_iz"])[0]-1
            plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
            plt.savefig("/home/mbk513/Documents/tanh-pump_particleSink_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
            plt.show()

    """ Particle Momentum """
    # params     = [dataDict["NVd+"]]#,dataDict["NVn"]]
    # colours    = ["tab:red"]#,      "tab:blue"]
    # markers    = [""]#,             ""]
    # labels     = ["NVd+"]#,"NVn"]
    # legend     = False
    # logPlot    = False
    # yLabel     = "Particle Mom."
    # startFrame = 0
    # frame      = -2
    # pause      = False
    # paramsPlot = [i[startFrame:,:] for i in params]
    # # paramsGif = gifParam(paramsPlot,cellPos,startFrame,logPlot,colours,markers,yLabel,legend,labels)
    # # plt.show()
    # if(stillFrames):
    #     stillFrame = np.shape(dataDict["NVd+"])[0]-6
    #     plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    #     plt.savefig("/home/mbk513/Documents/tanh-pump_particleMomentum_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
    #     stillFrame = np.shape(dataDict["NVd+"])[0]-1
    #     plotStillFrame(params,cellPos,stillFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    #     plt.savefig("/home/mbk513/Documents/tanh-pump_particleMomentum_frame-"+str(stillFrame),bbox_inches='tight',pad_inches=0.1)
    #     plt.show()

    """ Plasma Density """
    paramsPlot = [1e-19*dataDict["Nd"],1e-19*dataDict["Nd+"]]#,1e-19*dataDict["Ne"]]
    colours    = ["tab:red","tab:green"]#,"tab:blue"]
    markers    = ["",""]#,""]
    linestyles = ["-","--"]#,":"]
    labels     = ["Nd","Nd+"]#,"Ne"]
    legend     = True
    logPlot    = True
    yLabel     = "Plasma Density (1e19 m^-3)"
    startFrame = 0
    stopFrame  = -1#len(paramsPlot[0])#-500#
    frame      = -2
    pause      = False
    startXPos  = 0
    stopXPos   = -1
    paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in paramsPlot]
    paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # print("Minimum plasma density: %.3e"%(np.min(dataDict["Ne"])))
    plt.show()
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],1,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./densSlice_"+fileName+"timeIndex01.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],20,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./densSlice_"+fileName+"timeIndex20.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],60,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./densSlice_"+fileName+"timeIndex60.png",bbox_inches='tight',pad_inches=0.1)
    # plt.show()
    # f = r"./densAnimation.gif"
    # writergif = animation.PillowWriter(fps=10)
    # paramsGif.save(f, writer=writergif)
    # plt.show()

    # sys.exit()

    """ Plasma Temperature """
    paramsPlot = [dataDict["Td"],dataDict["Td+"],dataDict["Te"]]
    colours    = ["tab:red","tab:green","tab:blue"]
    markers    = ["","",""]
    linestyles = ["-","--",":"]
    labels     = ["Td","Td+","Te"]
    legend     = False
    logPlot    = False
    yLabel     = "T, (eV)"
    startFrame = 0
    frame      = -2
    pause      = False
    startXPos  = 0
    stopXPos   = -1
    paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in paramsPlot]
    # paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.show()
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],1,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./tempSlice_"+fileName+"timeIndex01.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],20,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./tempSlice_"+fileName+"timeIndex20.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],60,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./tempSlice_"+fileName+"timeIndex60.png",bbox_inches='tight',pad_inches=0.1)
    # plt.show()

    """ Plasma Pressure """
    paramsPlot = [dataDict["Pd"],dataDict["Pd+"],dataDict["Pe"]]
    colours    = ["tab:red","tab:green","tab:blue"]
    markers    = ["","",""]
    linestyles = ["-","--",":"]
    labels     = ["Pd","Pd+","Pe"]
    legend     = False
    logPlot    = False
    yLabel     = "P"
    startFrame = 0
    frame      = -2
    pause      = False
    startXPos  = 0
    stopXPos   = -1
    paramsPlot = [i[startFrame:stopFrame,startXPos:stopXPos] for i in paramsPlot]
    # paramsGif = gifParam(paramsPlot,cellPos[startXPos:stopXPos],startFrame,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.show()
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],1,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./presSlice_"+fileName+"timeIndex01.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],20,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./presSlice_"+fileName+"timeIndex20.png",bbox_inches='tight',pad_inches=0.1)
    # plotStillFrame(paramsPlot,cellPos[startXPos:stopXPos],60,logPlot,colours,markers,linestyles,yLabel,legend,labels)
    # plt.savefig("./presSlice_"+fileName+"timeIndex60.png",bbox_inches='tight',pad_inches=0.1)
    # plt.show()
    # # f = r"./tempAnimation.gif"
    # # writergif = animation.PillowWriter(fps=10)
    # # paramsGif.save(f, writer=writergif)
    # # plt.show()

# endFrame = 300
# dataDict["t_array"] = dataDict["t_array"][:endFrame]
# dataDict["Ne"]      = dataDict["Ne"][:endFrame,:]
# dataDict["P"]       = dataDict["P"][:endFrame,:]
# dataDict["Te"]      = dataDict["Te"][:endFrame,:]

# preTransInd = np.argmax(dataDict["t_array"])

if(0):
    """ #----- Plasma Density Evolution """
    plt.figure()
    plt.title("Plasma Density Evolution")
    plt.plot(dataDict["t_array"],dataDict["Ne"][:,-1],                                color="tab:red",   linestyle="-",  zorder=2,label="Ne[-1]")
    # plt.plot(dataDict["t_array"]*1e3,dataDict["Ne"][:,np.argmin(np.abs(cellPos-xPoint))], color="tab:green", linestyle="--",zorder=1,label="Ne[x-point]")
    plt.plot(dataDict["t_array"],dataDict["Ne"][:,0],                                 color="tab:orange",linestyle="--", zorder=3,label="Ne[0]")
    plt.plot(dataDict["t_array"],np.min(dataDict["Nd"][:,:],axis=1),                  color="tab:blue",  linestyle="-",  zorder=2,label="Nd[-1]")
    # plt.plot(dataDict["t_array"]*1e3,dataDict["Nd"][:,np.argmin(np.abs(cellPos-xPoint))], color="tab:orange",linestyle="-.",zorder=1,label="Nd[x-point]")
    plt.plot(dataDict["t_array"],dataDict["Nd"][:,0],                                 color="tab:cyan",  linestyle="--", zorder=3,label="Nd[0]")
    plt.plot(dataDict["t_array"],np.min(dataDict["Nd+"][:,:],axis=1),                 color="tab:green", linestyle="-",  zorder=2,label="Nd+[-1]")
    # plt.plot(dataDict["t_array"]*1e3,dataDict["Nd+"][:,np.argmin(np.abs(cellPos-xPoint))],color="tab:orange",linestyle="-.",zorder=1,label="Nd+[x-point]")
    plt.plot(dataDict["t_array"],dataDict["Nd+"][:,0],                                color="tab:olive", linestyle="--", zorder=3,label="Nd+[0]")
    plt.legend(loc="best")
    plt.xlabel("Time (ms)")
    plt.ylabel("Ne, t_array[:] (m^-3)")
    plt.tight_layout()

    """ #----- Target Pressure Evolution """
    plt.figure()
    plt.title("Target Pressure Evolution")
    plt.plot(dataDict["t_array"],dataDict["Pd"][:,-1],color="tab:green",linestyle="-",zorder=3,label="Pd[-1]")
    plt.legend(loc="best")
    plt.xlabel("Time (ms)")
    plt.ylabel("Target Pressure (Pa)")
    plt.tight_layout()

    """ #----- Summed Plasma Pressure Evolution """
    # plt.figure()
    # plt.title("Summed Plasma Pressure Evolution")
    # # plt.plot(dataDict["t_array"]*1e3,dataDict["P"][:,-1],                               color="tab:red",  linestyle="-", zorder=3,label="P[-1]")
    # # plt.plot(dataDict["t_array"]*1e3,dataDict["P"][:,np.argmin(np.abs(cellPos-xPoint))],color="tab:green",linestyle="--",zorder=1,label="P[x-point]")
    # # plt.plot(dataDict["t_array"]*1e3,dataDict["P"][:,0],                                color="tab:blue", linestyle="-", zorder=2,label="P[0]")
    # plt.plot(dataDict["t_array"],np.sum(dataDict["Pe"][:,:],axis=1),                               color="tab:red",  linestyle="-", zorder=3,label="P")
    # plt.plot(dataDict["t_array"],np.sum(dataDict["Pd"][:,:],axis=1),                               color="tab:green",  linestyle="-", zorder=3,label="P")
    # plt.plot(dataDict["t_array"],np.sum(dataDict["Pd+"][:,:],axis=1),                               color="tab:blue",  linestyle="-", zorder=3,label="P")
    # # plt.plot(dataDict["t_array"]*1e3,np.sum(dataDict["Pn"][:,:],axis=1),                               color="tab:blue",  linestyle="-", zorder=3,label="Pn")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("np.sum(P), t_array[:]")
    # plt.tight_layout()

    # """ #----- Total Particle Number Evolution """
    # plt.figure()
    # plt.title("Total Particle Number Evolution")
    # plt.plot(dataDict["t_array"]*1e3,np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1),color="tab:red",  linestyle="-",zorder=3,label="Ne")
    # plt.plot(dataDict["t_array"]*1e3,np.sum(dataDict["Nn"][:,1:-1]*cellVolume[1:-1],axis=1),color="tab:green",linestyle="-",zorder=3,label="Nn")
    # totalParticleNumber = np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)+np.sum(dataDict["Nn"][:,1:-1]*cellVolume[1:-1],axis=1)
    # plt.plot(dataDict["t_array"]*1e3,totalParticleNumber,                                color="tab:blue", linestyle="-",zorder=3,label="Tot")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("np.sum(Ne*cellVol)")
    # plt.tight_layout()

    """ #----- Temperature Evolution """
    plt.figure()
    plt.title("Temperature Evolution")
    plt.plot(dataDict["t_array"][1:],dataDict["Te"][1:,-1], color="tab:red",   linestyle="-", zorder=2,label="Te[-1]")
    plt.plot(dataDict["t_array"][1:],dataDict["Td+"][1:,-1],color="tab:blue",  linestyle="-", zorder=2,label="Td+[-1]")
    plt.plot(dataDict["t_array"][1:],dataDict["Te"][1:,0],  color="tab:brown", linestyle=":", zorder=3,label="Te[0]")
    plt.plot(dataDict["t_array"][1:],dataDict["Td+"][1:,0], color="tab:cyan",  linestyle=":", zorder=3,label="Td+[0]")
    plt.plot(dataDict["t_array"][1:],dataDict["Td"][1:,-1], color="tab:green", linestyle="-", zorder=2,label="Td[-1]")
    plt.plot(dataDict["t_array"][1:],dataDict["Td"][1:,0],  color="tab:olive", linestyle=":", zorder=3,label="Td[0]")
    plt.legend(loc="best")
    plt.xlabel("Time (ms)")
    plt.ylabel("Te, t_array[:] (eV)")
    plt.tight_layout()

    # """ #----- Total Particle Number Evolution """
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    # totalParticleNumber = np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)+np.sum(dataDict["Nn"][:,1:-1]*cellVolume[1:-1],axis=1)
    # plt.title("Total Particle Number Evolution")
    # ax1.plot(dataDict["t_array"]*1e3,1e-19*np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)/np.sum(cellVolume[1:-1]),color="tab:red",  linestyle="-",zorder=3,label="Ne")
    # ax1.plot(dataDict["t_array"]*1e3,1e-19*np.sum(dataDict["Nn"][:,1:-1]*cellVolume[1:-1],axis=1)/np.sum(cellVolume[1:-1]),color="tab:green",linestyle="-",zorder=3,label="Nn")
    # ax1.plot(dataDict["t_array"]*1e3,1e-19*totalParticleNumber/np.sum(cellVolume[1:-1]),                                   color="tab:blue", linestyle="-",zorder=3,label="Tot")
    # ax2.plot(tind,1e-19*totalParticleNumber/np.sum(cellVolume[1:-1]),color="white",alpha=0)
    # ax1.set_zorder(1)
    # ax2.set_zorder(-1)
    # ax1.legend(loc="best")
    # ax1.set_xlabel("Sim. Time (ms)")
    # ax2.set_xlabel("Sim. tind")
    # ax1.set_ylabel("Mean Density (1e19 m^-3)")
    # # plt.show()

    """ #----- Final Density Profiles """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    preTransInd = 0
    plt.title("Final Density Profiles")
    ax1.plot(cellPos[:],dataDict["Ne"][-1,:],         linestyle="-", marker="",color="tab:red",   label="Electrons",zorder=1)
    ax1.plot(cellPos[:],dataDict["Nd"][-1,:],         linestyle="--",marker="",color="tab:green", label="Neutrals", zorder=2)
    ax1.plot(cellPos[:],dataDict["Nd+"][-1,:],        linestyle=":", marker="",color="tab:blue",  label="Ions",     zorder=2)
    ax1.plot(cellPos[:],dataDict["Ne"][preTransInd,:],linestyle="-", marker="",color="tab:brown", label="pre-Elecs",zorder=0)
    ax1.plot(cellPos[:],dataDict["Nd"][preTransInd,:],linestyle="--",marker="",color="tab:olive", label="pre-Neuts",zorder=0)
    ax1.plot(cellPos[:],dataDict["Nd+"][preTransInd,:],linestyle=":",marker="",color="tab:purple",label="pre-Ions", zorder=0)
    ax1.axhline(y=Nnorm*1e-5,linestyle="--",color="tab:grey")
    ax1.legend(loc="best")
    ax1.set_xlabel("Cell Posistion (m)")
    ax1.set_ylabel(r"Particle Density (m$^-3$)")
    ax1.set_yscale("log")
    ax1.set_zorder(2)
    if(pump):
        ax2 = ax1.twinx()
        ax2.plot(cellPos[1:-1],pumpRegion[1:-1],color="tab:grey",linestyle="--")
        ax2.set_ylabel("Pump Location",color="tab:grey")
        ax2.tick_params(labelcolor="tab:grey")
        ax2.axvline(x=63,color="black",linestyle="--")
        ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    plt.tight_layout()
    print("Upstream density = %.3e m^-3"%dataDict["Ne"][-1,0])
    # totalPlasmaNumber = np.sum(dataDict["Ne"][:,1:-1]*cellVolume[1:-1],axis=1)
    # meanPlasmaDensity = totalPlasmaNumber/np.sum(cellVolume[1:-1])
    # print(meanPlasmaDensity)

    """ #----- Final Temperature Profiles """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title("Final Temperature Profile")
    ax1.plot(cellPos[:],dataDict["Te"][-1,:],          linestyle="-", color="tab:red",   zorder=2,marker="", label="Te")
    ax1.plot(cellPos[:],dataDict["Td"][-1,:],          linestyle="--",color="tab:green", zorder=2,marker="", label="Td")
    ax1.plot(cellPos[:],dataDict["Td+"][-1,:],         linestyle=":", color="tab:blue",  zorder=2,marker="", label="Td+")
    ax1.plot(cellPos[:],dataDict["Te"][preTransInd,:], linestyle="-", color="tab:brown", zorder=1,marker="", label="pre-Te")
    ax1.plot(cellPos[:],dataDict["Td"][preTransInd,:], linestyle="--",color="tab:olive", zorder=1,marker="", label="pre-Td")
    ax1.plot(cellPos[:],dataDict["Td+"][preTransInd,:],linestyle=":", color="tab:purple",zorder=1,marker="", label="pre-Td+")
    # plt.axvline(x=76.13,color="tab:grey",linestyle="--")
    # plt.axvline(x=77.13,color="tab:grey",linestyle="--")
    ax1.legend(loc="best")
    # plt.xlim([78.0,78.140])
    # plt.ylim([0,16])
    # plt.axhline(y=5,linestyle="--",color="tab:grey")
    # plt.axvline(x=78.067,linestyle="--",color="tab:blue")
    # plt.xlabel("cellPos (m)")
    ax1.set_xlabel("Cell Posistion (m)")
    # plt.ylabel("Te[:], t_array[-1] (eV)")
    ax1.set_ylabel("Temperature (eV)")
    ax1.set_yscale("log")
    ax1.set_zorder(2)
    if(pump):
        ax2 = ax1.twinx()
        ax2.plot(cellPos[1:-1],pumpRegion[1:-1],color="tab:grey",linestyle="--")
        ax2.set_ylabel("Pump Location",color="tab:grey")
        ax2.tick_params(labelcolor="tab:grey")
        ax2.axvline(x=63,color="black",linestyle="--")
        ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    plt.tight_layout()

    """ #----- Final Pressure Profiles """
    plt.figure()
    plt.title("Final Pressure Profile")
    plt.plot(cellPos[:],dataDict["Pe"][-1,:],          linestyle="-", color="tab:red",   zorder=2,label="Pe")
    plt.plot(cellPos[:],dataDict["Pd"][-1,:],          linestyle="--",color="tab:green", zorder=2,label="Pd")
    plt.plot(cellPos[:],dataDict["Pd"][0,:],           linestyle="--",color="tab:green", zorder=2,label="Pd[t=0]")
    plt.plot(cellPos[:],dataDict["Pd+"][-1,:],         linestyle=":", color="tab:blue",  zorder=2,label="Pd+")
    plt.plot(cellPos[:],dataDict["Pe"][preTransInd,:], linestyle="-", color="tab:brown", zorder=2,label="pre-Pe")
    plt.plot(cellPos[:],dataDict["Pd"][preTransInd,:], linestyle="--",color="tab:olive", zorder=2,label="pre-Pd")
    plt.plot(cellPos[:],dataDict["Pd+"][preTransInd,:],linestyle=":", color="tab:purple",zorder=2,label="pre-Pd+")
    # plt.plot(cellPos[:],np.gradient(dataDict["P"][-1,:],cellPos[:]), color="tab:red",zorder=2,label="ddx")
    # plt.plot(cellPos[:],dataDict["P"][preTransInd,:],color="tab:cyan",zorder=1,label="pre-trans",linestyle="--")
    plt.axhline(y=Nnorm*1e-5/(1.602e19),linestyle="--",color="tab:grey")
    print("Target Pressure  = %.3e Pa"%dataDict["Pd"][-1,-1])
    plt.legend(loc="best")
    plt.xlabel("cellPos (m)")
    plt.ylabel("Pressure (Pa)")
    plt.tight_layout()

    """ #----- Final Momentum Profile """
    # plt.figure()
    # plt.title("Final Momentum Profile")
    # plt.plot(cellPos[:],dataDict["NVi"][-1,:], color="tab:blue",zorder=2,label="NVi")
    # plt.plot(cellPos[:],dataDict["NVn"][-1,:],color="tab:green",zorder=2,label="NVn")
    # # plt.plot(cellPos[:],np.gradient(dataDict["P"][-1,:],cellPos[:]), color="tab:red",zorder=2,label="ddx")
    # # plt.plot(cellPos[:],dataDict["P"][preTransInd,:],color="tab:cyan",zorder=1,label="pre-trans",linestyle="--")
    # plt.legend(loc="upper left")
    # plt.xlabel("cellPos (m)")
    # plt.ylabel("NVx[:], t_array[-1]")
    # plt.tight_layout()

""" #----- Det. Point Movement """
fig = plt.figure()
ax1 = fig.add_subplot(111)
fig.suptitle("Det. Point Movement")
detLoc    = []
tempVal   = 7.0 # eV
tempMean  = (dataDict["Te"]+dataDict["Td+"])/2.
detLoc    = cellPos[-1]-cellPos[np.argmin(np.abs(tempMean-tempVal),axis=1)]
print("Front position   = %.3e m"%detLoc[-1])
#----- Saves the detLoc vs udens
if(0):
    df = pd.DataFrame({"t_array":dataDict["t_array"][1:],"detLoc":detLoc[1:],"udens":dataDict["Nd+"][1:,0]})
    df.to_csv("./misc/detFront_Mvmt_"+filePath.split("/")[3]+".csv")
#----- Expo decay to target
if(0):
    from scipy.optimize import curve_fit
    argBounce = np.where(detLoc==np.min(detLoc))[0][0]
    xTime     = np.copy(dataDict["t_array"])[:argBounce]
    detLocFit = np.copy(detLoc)[:argBounce]
    detLocFit = detLocFit[xTime>=0.0]
    xTime     = xTime[xTime>=0.0]

    xTime     = xTime[detLocFit>=3.0]
    detLocFit = detLocFit[detLocFit>=3.0]

    guess     = [np.max(detLocFit),np.max([10.*(options["Pe"]["powFactor"]-1.0)*0.772*1e-3,1.0])]
    def expo(t,A,k):
        return A*np.exp(-k*t)
    popt, pcov = curve_fit(expo,xTime,detLocFit,p0=guess,bounds=[[-0.0001+np.max(detLocFit)*0.999,0.],[0.0001+np.max(detLocFit)*1.001,np.inf]])
    # ax1.plot(xTime,1e2*expo(xTime,*guess),linestyle="--",color="tab:orange",label="SH Fit")
    ax1.plot(xTime,1e2*expo(xTime,*popt), linestyle="--", color="tab:brown", label="MK Fit")
#----- Rebound back out
if(0):
    from scipy.optimize import curve_fit
    argBounce = np.where(detLoc==np.min(detLoc))[0][-1]
    xTime     = np.copy(dataDict["t_array"])[argBounce:]
    minTime   = xTime[0]
    xTime    -= minTime
    detLocFit = np.copy(detLoc)[argBounce:]
    guess     = [9.9985,np.min(detLocFit),5e-3]
    def oneMinusExpo(t,yMax,yMin,l):
        return yMax-(yMax-yMin)*np.exp(-l*t)
    lB, uB = [0.,np.min(detLocFit)*0.9,0.], [100,0.0001+np.min(detLocFit)*1.1,np.inf]
    popt, pcov = curve_fit(oneMinusExpo,xTime,detLocFit,p0=guess,bounds=(lB,uB))
    ax1.plot(xTime+minTime, 1e2*oneMinusExpo(xTime,*popt), linestyle="--", color="tab:purple",label="rebound")
ax1.plot(dataDict["t_array"][1:],1e2*np.array(detLoc[1:]),color="tab:red")#,marker="x")
ax1.legend(loc="best")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Detachment Front (cm)")
ax1.set_zorder(1)
ax1.patch.set_visible(False)
#----- ax2 Pow Factor
# ax2 = ax1.twinx()
# # ax2.plot(dataDict["t_array"][1:],np.sum(dataDict["Pe_src"],axis=1)[1:]/np.min(np.sum(dataDict["Pe_src"],axis=1)[1:]),color="black")
# ax2.plot(dataDict["t_array"][1:],np.sum(dataDict["Pe_src"],axis=1)[1:],color="black")
# # ax2.set_ylim([0,3])
# ax2.set_ylabel("Power Factor")
# ax2.set_zorder(0)
plt.tight_layout()

if(0):
    """ #----- Upstream Density """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    # fig.suptitle("Upstream Density")
    ax2.plot(dataDict["Nd+"][1:,0]/1e19,color="white",alpha=0)
    ax1.plot(dataDict["t_array"][1:],dataDict["Nd+"][1:,0]/1e19,color="tab:red")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel(r"Upstream Plasma Density [10$^{19}$ m$^{-1}$]")
    ax2.set_xlabel("Sim. tind")
    plt.tight_layout()

    """ #----- Domain Filling """
    plt.figure()
    plt.title("Domain Filling")
    totalPlasmaNumber   = np.sum(dataDict["Nd+"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalNeutralNumber  = np.sum(dataDict["Nd"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalParticleNumber = totalNeutralNumber+totalPlasmaNumber
    meanParticleNumber  = totalParticleNumber/np.sum(cellVolume[1:-1])
    plt.plot(meanParticleNumber[1:]/1e19,dataDict["Nd+"][1:,0]/1e19,color="tab:purple")
    plt.xlabel("Mean density (1e19)")
    plt.ylabel("Upstream Density (1e19)")
    plt.tight_layout()

    """ #----- Det. Pos. vs Filling """
    plt.figure()
    plt.title("Det. Pos. vs Filling")
    totalPlasmaNumber   = np.sum(dataDict["Nd+"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalNeutralNumber  = np.sum(dataDict["Nd"][:,1:-1]*cellVolume[1:-1],axis=1)
    totalParticleNumber = totalNeutralNumber+totalPlasmaNumber
    meanParticleNumber  = totalParticleNumber/np.sum(cellVolume[1:-1])
    plt.plot(meanParticleNumber[1:]/1e19,1e2*np.array(detLoc[1:]),color="tab:orange")
    plt.ylabel("Detachment Front Position (cm)")
    plt.xlabel("Mean Particle Number (1e19)")
    plt.tight_layout()

    """ #----- Udens vs Det. Pos. """
    plt.figure()
    mpl.rcParams['font.size']=16
    if(0):
        div1d_mvmt = pd.read_csv("./misc/DIV1D-NN_detFront_vs_udens.csv")
        plt.plot(div1d_mvmt["udens"],div1d_mvmt["detFront"],linewidth=2,color="tab:blue",label="DIV1D-NN")
        plt.xlim([1.0,5.0])
        plt.ylim([-0.1,5])
    # plt.xlim([1.0,3.0])
    # plt.ylim([-0.1,10])
    plt.plot(dataDict["Nd+"][1:,0]/1e19,np.array(detLoc[1:]),linewidth=2,color="tab:red",label="Hermes-3")
    plt.legend(loc="best")
    # plt.xlabel(r"Upstream Plasma Density [10$^{19}$ m$^{-1}$]")
    plt.xlabel(r"udens / udens${_0}$")
    plt.ylabel("Front Distance to Target [m]")
    plt.grid(True)
    plt.tight_layout()

    """ #----- DIV1D Copy """
    # xPt = np.argmin(np.abs(cellPos-xPoint))
    # cellPos = 7. - cellPos
    # cellPos = cellPos[xPt:]
    # mpl.rcParams['font.size']=10
    # # plt.subplot(5,1,1)
    # DIV1D_q_par = pd.read_csv("./misc/DIV1D_q_par.csv")
    # plt.plot(DIV1D_q_par["xPos"],DIV1D_q_par["q_par"], linewidth=2,color="tab:blue",label="DIV1D")
    # plt.plot(cellPos,            np.ones(len(cellPos)),linewidth=2,color="tab:red", label="Hermes-3")
    # plt.xlim([5,0])
    # plt.legend(loc="upper right")
    # plt.ylabel(r"q$_{||}$ (MW m$^{-2}$")
    # plt.grid(True)
    # plt.show()
    #
    # # plt.subplot(5,1,2)
    # DIV1D_Te = pd.read_csv("./misc/DIV1D_Te.csv")
    # plt.plot(DIV1D_Te["xPos"],DIV1D_Te["Te"],                                        linewidth=2,color="tab:blue",  label="DIV1D")
    # plt.plot(cellPos,         dataDict["Te"][-1,xPt:], linewidth=2,color="tab:red",   label="Hermes-3 (e)")
    # plt.plot(cellPos,         dataDict["Td+"][-1,xPt:],linewidth=2,color="tab:orange",label="Hermes-3 (d+)")
    # plt.xlim([5,0])
    # plt.legend(loc="upper right")
    # plt.ylabel("T (eV)")
    # plt.grid(True)
    # plt.show()
    #
    # # plt.subplot(5,1,3)
    # DIV1D_Ne = pd.read_csv("./misc/DIV1D_Ne.csv")
    # plt.plot(DIV1D_Ne["xPos"],DIV1D_Ne["Ne"],                                             linewidth=2,color="tab:blue",label="DIV1D")
    # plt.plot(cellPos,         dataDict["Ne"][-1,xPt:]*1e-19,linewidth=2,color="tab:red", label="Hermes-3")
    # plt.xlim([5.,0.])
    # # plt.ylim([1.,4.])
    # plt.legend(loc="upper right")
    # plt.ylabel(r"n$_e$ (m$^{-3}$)")
    # plt.yscale("log")
    # plt.grid(True)
    # plt.show()
    #
    # # plt.subplot(5,1,4)
    # DIV1D_v_par = pd.read_csv("./misc/DIV1D_v_par.csv")
    # plt.plot(DIV1D_v_par["xPos"],DIV1D_v_par["v_par"],                                     linewidth=2,color="tab:blue",  label="DIV1D")
    # plt.plot(cellPos,            dataDict["Ve"][-1,xPt:]*1e3,linewidth=2,color="tab:red",   label="Hermes-3")
    # plt.xlim([5,0])
    # # plt.ylim([0,30])
    # plt.legend(loc="upper right")
    # plt.ylabel(r"v$_{||}$ (10$^3$ms$^{-1}$)")
    # plt.grid(True)
    # plt.show()
    #
    # # plt.subplot(5,1,5)
    # DIV1D_Nd = pd.read_csv("./misc/DIV1D_Nd.csv")
    # plt.plot(DIV1D_Nd["xPos"],DIV1D_Nd["Nd"],                                             linewidth=2,color="tab:blue",label="DIV1D")
    # plt.plot(cellPos,         dataDict["Nd"][-1,xPt:]*1e-18,linewidth=2,color="tab:red", label="Hermes-3")
    # plt.xlim([5,0])
    # plt.legend(loc="upper right")
    # plt.ylabel(r"n$_{D0}$ (m$^{-3}$)")
    # plt.yscale("log")
    # plt.grid(True)
    ##################

plt.show()
sys.exit()
