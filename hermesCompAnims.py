# from boututils.showdata import showdata
from scipy.optimize import curve_fit
from boutdata.data import BoutData
import matplotlib.pyplot as plt
from boutdata import collect
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import sys

# This in-place replaces the points in the guard cells with the points on the boundary
def replace_guards(var):
    var[0]  = 0.5*(var[0] + var[1])
    var[-1] = 0.5*(var[-1] + var[-2])
    return var

# Load in all the relevant data
def loadData(filePath):
    boutdata = BoutData(filePath, yguards=True, strict = True, DataFileCaching=False)
    options  = boutdata["options"]

    """ Path to the directory containing the data """
    dataDict = dict()
    dataDict["J"] = collect("J",path=filePath,yguards=True,info=False,strict=True)[0,1:-1]

    # Make the cell position grid
    xPoint     = options["mesh"]["length_xpt"]
    dy         = collect("dy",path=filePath,yguards=True,info=False,strict=True)[0,1:-1]
    cellPos    = np.zeros(len(dy))
    cellPos[0] = -0.5*dy[1]
    cellPos[1] =  0.5*dy[1]
    for i in range(2,len(dy)):
        cellPos[i] = cellPos[i-1] + 0.5*dy[i-1] + 0.5*dy[i]
    dataDict["cellPos"] = cellPos
    dataDict["xPoint"]  = xPoint

    # Normalisation factors
    Nnorm    = collect("Nnorm",   path=filePath)
    Cs0      = collect("Cs0",     path=filePath)
    rho_s0   = collect("rho_s0",  path=filePath)
    Omega_ci = collect("Omega_ci",path=filePath)
    Tnorm    = collect("Tnorm",   path=filePath)
    Pnorm    = Nnorm*Tnorm*1.602e-19 # Converts p to Pascals

    # Cell volume is the volume of each cell, stepTime is the real time of each output step
    # J must be normalised by rho_s0^2 as it is effectively an area and x,y=1
    if not(isinstance(options["timestep"], int)):
        try:
            options["timestep"] = eval(options["timestep"])
        except eE as Exception:
            print(eE)
            print("Timestep error")
    cellVolume = dataDict["J"]*rho_s0*rho_s0*dy
    stepTime = options["timestep"]/Omega_ci
    dataDict["cellVolume"] = cellVolume

    # Timing data
    dataDict["t_array"]    = collect("t_array", path=filePath,yguards=True,info=False,strict=True)
    dataDict["iteration"]  = collect("t_array", path=filePath,yguards=True,info=False,strict=True)
    dataDict["wtime"]      = collect("wtime",   path=filePath,yguards=True,info=False,strict=True)
    dataDict["wall_time"]  = collect("wall_time",path=filePath,yguards=True,info=False,strict=True)
    dataDict["ncalls"]     = collect("ncalls",  path=filePath,yguards=True,info=False,strict=True)

    # Normalise to time array
    tind = np.copy(dataDict["t_array"]/options["timestep"])
    dataDict["t_array"] /= Omega_ci # stepTime/options["timestep"]
    dataDict["t_array"] -= dataDict["t_array"][0]
    dataDict["t_array"] *= 1e3

    """ Load all variables """
    dataDict["Nd"]      = collect("Nd",    path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Nd+"]     = collect("Nd+",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Ne"]      = collect("Ne",    path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    dataDict["Pd"]       = collect("Pd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Pd+"]      = collect("Pd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Pe"]       = collect("Pe",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    dataDict["Td"]       = collect("Td",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Td+"]      = collect("Td+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    dataDict["Te"]       = collect("Te",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    try:
        dataDict["Rimp"] = collect("Rar", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    except:
        try:
            dataDict["Rimp"] = collect("Rc", path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
        except:
            print("Couldn't find impurity radiation!")
            dataDict["Rimp"] = np.zeroes(len(dataDict["t_array"]))

    # # dataDict["NVd"]       = collect("NVd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    # dataDict["NVd+"]      = collect("NVd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    # """ NVe == NVd+ """
    # dataDict["NVe"]       = collect("NVd+",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    # # Shape is [0,time,0]???
    # # dataDict["Vd"]       = collect("Vd",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    # dataDict["Vd+"]      = collect("Vd+",  path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]
    # dataDict["Ve"]       = collect("Ve",   path=filePath,yguards=True,info=False,strict=True)[:,0,1:-1,0]

    atomics = False
    if(atomics):
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

    # # dataDict["NVd"]  *= Cs0*Nnorm
    # dataDict["NVd+"] *= Cs0*Nnorm
    # dataDict["NVe"]  *= Cs0*Nnorm
    #
    # # dataDict["Vd"]   *= dataDict["NVd"]/dataDict["Nd"]
    # dataDict["Vd+"]  *= dataDict["NVd+"]/dataDict["Nd+"]
    # dataDict["Ve"]   *= dataDict["NVe"]/dataDict["Ne"]

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
            dataDict["Rimp"][tind,:]    = replace_guards(dataDict["Rimp"][tind,:])
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

    return dataDict

#############################################
## Load in the shot data and input options ##
#############################################

#----- Old Runs
if(0):
    """ STEP Power Drop Scan """
    # path = "./simulations/WP_X/"
    # extens = ["1-2/","1-4/","1-8/","1-16/"]
    # fileNames = [path+"spr45_powerScan_baseline_"+exten for exten in extens]

    """ Flat STEP Power Drop Scan """
    # path = "./simulations/WP_X/"
    # extens = ["1-2/","1-4/","1-8/","1-16/","1-32/","1-64/"]
    # fileNames = [path+"spr45_powerScan_flat_"+exten for exten in extens]

    """ Real J STEP Power Drop Scan """
    # path = "./simulations/WP_X/"
    # extens = ["1-2/","1-4/","1-8/","1-16/","1-32/","1-64_RA/"]
    # fileNames = [path+"spr45_powerScan_realJ_"+exten for exten in extens]

    """ Real J Neutral thermal_conduction STEP Power Drop Scan """
    # path = "./simulations/WP_X/"
    # # extens = ["1-2/","1-8/","1-32/"]
    # # fileNames = [path+"spr45_powerScan_realJ_neutCond_"+exten for exten in extens]
    # extens = ["1-2/","neutCond_1-2/","neutCond_1-8/","1-8/","neutCond_1-32/","1-32/"]
    # fileNames = [path+"spr45_powerScan_realJ_"+exten for exten in extens]

    """ Real J Neutral thermal_conduction STEP Power Drop Scan """
    # path = "./simulations/WP_X/"
    # extens = ["1-2/","1-4/","1-8/"]
    # fileNames = [path+"spr45_powerScan_realJ_fimp04_"+exten for exten in extens]

    """ Real J Weak det. STEP Power Increase Scan """
    # path = "./simulations/WP_X/"
    # extens = ["2x/","4x/","8x/"]
    # fileNames = [path+"spr45_powerScan_realJ_fimp04_"+exten for exten in extens]

    """ Set up for Nitrogen low det 1m """
    # path = "./simulations/WP_X/"
    # extens = ["lowDet/","lowDet_II/","lowDet_III/"]
    # fileNames = [path+"spr45_realJ_neutMom_fimp04_nitrogen_"+exten for exten in extens]

    """ Real J Weak det. Neutral Momentum STEP Power Increase Scan """
    # path = "./simulations/WP_X/"
    # extens = ["2x/","4x/","8x/"]
    # fileNames = [path+"spr45_realJ_neutMom_fimp04_lowDet_"+exten for exten in extens]

    """ Comparing Real J Weak det. with Neutral Momentum STEP Power Increase Scan """
    # path = "./simulations/WP_X/"
    # extens = ["realJ_neutMom_fimp04_lowDet_2x/","powerScan_realJ_fimp04_2x/","realJ_neutMom_fimp04_lowDet_4x/","powerScan_realJ_fimp04_4x/","realJ_neutMom_fimp04_lowDet_8x/","powerScan_realJ_fimp04_8x/"]
    # fileNames = [path+"spr45_"+exten for exten in extens]

    """ Real J Weak det. with Neutral Momentum and Nitrogen impurity STEP Power Increase Scan """
    # path = "./simulations/WP_X/"
    # extens = ["2x/","4x/","8x/"]
    # fileNames = [path+"spr45_realJ_neutMom_fimp04_nitrogen_lowDet_2-0_"+exten for exten in extens]

    """ Real J Neutral Diffusion 7e19 starting density, increaseing impurity fraction """
    # path = "./simulations/WP_X/"
    # extens = ["fimp02", "fimp04", "fimp08"]
    # fileNames = [path+"spr45_realJ_neutDiff_"+exten+"_07e19" for exten in extens]

    """ Real J Neutral Diffusion 20e19 starting density, increaseing impurity fraction """
    # path = "./simulations/WP_X/"
    # extens = ["fimp02", "fimp04", "fimp08"]
    # fileNames = [path+"spr45_realJ_neutDiff_"+exten+"_20e19" for exten in extens]

    """ Real J Neutral Diffusion 28e19 starting density, changing the neutral diffusion coefficient """
    # path = "./simulations/WP_X/"
    # extens = ["_dneut05", "_dneut20", "_dneut40"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_28e19"+exten for exten in extens]
    # labels = ["dneut =  5", "dneut = 20", "dneut = 40"]

    """ Real J Neutral Diffusion 40e19 starting density, changing the neutral diffusion coefficient """
    # path = "./simulations/WP_X/"
    # extens = ["dneut05", "extended", "dneut20", "dneut40"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_40e19_"+exten for exten in extens]
    # # labels = ["dneut =  5", "dneut = 10", "dneut = 20", "dneut = 40"]

    """ Restart from fimp01 40e19 case with different fimps """
    # path = "./simulations/WP_X/"
    # extens = ["fimp01_40e19_extended", "fimp02_40e19", "fimp04_40e19", "fimp08_40e19"]
    # fileNames = [path+"spr45_realJ_neutDiff_"+exten for exten in extens]
    # labels = ["imp frac = 1%", "imp frac = 2%", "imp frac = 4%", "imp frac = 8%"]

    """ Real J Neutral Diffusion, 1% impurity fraction, 40e19 startign density increasing starting densities """
    # path = "./simulations/WP_X/"
    # extens = ["20e19", "32e19", "34e19", "36e19", "38e19", "40e19", "42e19", "44e19","46e19", "48e19", "50e19", "52e19"]#]#
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_"+exten for exten in extens]

    """ Restart from fimp01 40e19 case with different fimps """
    # path = "./simulations/WP_X/"
    # extens = ["_pow_2-000x", "_pow_1-414x", "_extended", "_pow_0-707x", "_pow_0-500x"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_40e19"+exten for exten in extens]
    # labels = ["pow*2", "pow*sqrt(2)", "pow", "pow/sqrt(2)", "pow/2"]

    """ Real J, 40e19, 1% fimp, turning on and off neutral momentum """
    # path = "./simulations/WP_X/"
    # extens = ["neutDiff", "neutMom"]
    # fileNames = [path+"spr45_realJ_"+exten+"_fimp01_40e19/" for exten in extens]

    """ CE comp """
    # path = "./simulations/WP_X/"
    # extens = ["28e19", "28e19_noCE_collisions"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_"+exten for exten in extens]
    # extens = ["40e19_initial", "40e19_noCE_collisions"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_"+exten for exten in extens]

    """ SS detachment 15m away from target, power step up, different dneut """
    # path = "./simulations/WP_X/"
    # extens = ["15m_dneut10_pow2x","15m_dneut10_pow4x","15m_dneut32_pow2x","15m_dneut32_pow4x","15m_dneut100_pow2x","15m_dneut100_pow4x"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_28e19_"+exten for exten in extens]
    # # labels = ["pow*2", "pow*sqrt(2)", "pow", "pow/sqrt(2)", "pow/2"]

    """ SS detachments away from target, power step up, different dneut """
    # path = "./simulations/WP_X/"
    # extens = ["05m_dneut10_pow2x","05m_dneut10_pow4x","10m_dneut10_pow2x","10m_dneut10_pow4x","15m_dneut10_pow2x","15m_dneut10_pow4x"]
    # fileNames = [path+"spr45_realJ_neutDiff_fimp01_28e19_"+exten for exten in extens]
    # # labels = ["pow*2", "pow*sqrt(2)", "pow", "pow/sqrt(2)", "pow/2"]

    """ TCV flux limiter comparison """
    # path = "./simulations/WP_X/"
    # extens = ["fluxLim","noFluxLim"]
    # fileNames = [path+"TCV_22MW_3pc-carbon_longUpstream_"+exten for exten in extens]

    """ TCV Changing fimp """
    # path = "./simulations/WP_X/"
    # extens = ["1","5","7"]
    # fileNames = [path+"TCV_22MW_"+exten+"pc-carbon_longUpstream" for exten in extens]

    """ TCV Changing Power """
    # path = "./simulations/WP_X/"
    # extens = ["11","16","31"]
    # fileNames = [path+"TCV_"+exten+"MW_3pc-carbon_longUpstream" for exten in extens]

    """ TCV powScan """
    # path = "./simulations/WP_X/"
    # extens = ["15-6","11-0","07-8","05-5","03-9","02-8","01-9","01-4","01-0","00-7","00-5","00-3"]
    # fileNames = [path+"TCV_22MW_3pc-carbon_longUpstream_powScan_"+exten+"MW" for exten in extens]

    """ TCV fimpScan """
    # path = "./simulations/WP_X/"
    # extens = ["04-2","06-0","08-5","12-0","17-0","24-0","33-9","48-0"]
    # fileNames = [path+"TCV_22MW_3pc-carbon_longUpstream_fimpScan_"+exten for exten in extens]

    """ TCV 2e19 power scan """
    # path = "./simulations/WP_X/"
    # extens = ["2_0","1_4","1-2","1-4","1-8","1-16","1-32","1-64"]
    # fileNames = [path+"TCV_22MW_3pc-carbon_longUpstream_2e19udens_powScan_"+exten for exten in extens]

    """ TCV 2e19 fimp scan """
    # path = "./simulations/WP_X/"
    # extens = ["01","02","05","08","11","15","20","26"]
    # fileNames = [path+"TCV_22MW_3pc-carbon_longUpstream_2e19udens_fimpScan_"+exten for exten in extens]

    """ MAST-U 1.7e19 PI power scan """
    # path = "./simulations/WP_X/"
    # # extens = ["8_0","4_0","2_0","1_59","1_26","2-5","9-10","1-2","1-4","1-8","1-16","1-32"]
    # extens = ["2_0","1_59","1_26","1-2","1-4","9-10","2-5"]
    # fileNames = [path+"MAST-U_1-7e19udens_PI_powScan_"+exten for exten in extens]

    """ MAST-U 1.7e19 power scan """
    # path = "./simulations/WP_X/"
    # extens = ["8_0","4_0","2_0","1-2","1-4","1-8","1-16","1-32"]
    # fileNames = [path+"MAST-U_1-7e19udens_powScan_"+exten for exten in extens]

    """ MAST-U 1.7e19 fimp scan """
    # path = "/home/mbk513/Downloads/" # "./simulations/WP_X/"
    # extens = ["01","02","03-5","08","11","15","20","25"]
    # fileNames = [path+"MAST-U_1-7e19udens_fimpScan_"+exten for exten in extens]

    """ MAST-U 1.7e19 PI fimp scan """
    # path = "./simulations/WP_X/"
    # extens = ["02-5","06","07","08","09","10","15","20"]
    # fileNames = [path+"MAST-U_1-7e19udens_PI_fimpScan_"+exten for exten in extens]

    """ MAST-U 1.7e19 dens scan """
    # path = "./simulations/WP_X/"
    # extens = ["1-75","2-00","2-25","2-50","2-75"]
    # fileNames = [path+"MAST-U_1-7e19udens_densScan_"+exten+"e19" for exten in extens]

    """ STEP 40 m Power Scan """
    # path =  "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # extens = ["2"]#,"4","8","16","32","64","128","256"]
    # fileNames = [path+"spr45_realJ_40m_powScan_"+exten for exten in extens]

    """ STEP 40 m PI Power Scan"""
    # path = "./simulations/WP_X/" #
    # extens = ["2","4","8","16"]
    # fileNames = [path+"spr45_realJ_40m_PI_powScan_"+exten for exten in extens]

    """ STEP 10 m power step vs pulse """
    # path = "/home/mbk513/Downloads/" # "./simulations/WP_X/" #
    # extens = ["4x","4x_pulse"]
    # fileNames = [path+"spr45_realJ_10m_det_"+exten for exten in extens]

    """ STEP 10 m power increases """
    # path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # extens = ["1-490x"]
    # # extens = ["1-350x","1-360x","1-375x","1-385x","1-400x","1-414x","1-430x","1-445x","1-460x","1-490x","1-600x","1-682x","1-850x","2-000x"]
    # # extens = ["1-189x","1-200x","1-250x","1-350x","1-360x","1-375x","1-385x","1-400x","1-414x","1-430x","1-445x","1-460x","1-490x","1-600x","1-682x","1-850x","2-000x"]# "1-189x_fL-MC",
    # fileNames = [path+"spr45_realJ_10m_det_PI_powScan_"+exten for exten in extens]
    # labels = [float(ext.replace("-",".").replace("x","")) for ext in extens]

    """ STEP 10 m power increases - hysteresis """
    # path =  "./simulations/WP_X/"
    # extens = ["1-350x","1-414x","1-600x","1-850x"]
    # fileNames = [path+"spr45_realJ_10m_det_PI_powScan_"+exten+"-hyst" for exten in extens]
    # labels = [1.350,1.414,1.600,1.850]

    """ STEP 10 m power pulses """
    # path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # # extens = ["1ms_2x","1ms_4x","10ms_2x","10ms_4x","100ms_2x","100ms_4x"]
    # # extens = ["10ms_2x","10ms_4x"]
    # extens = ["100ms_2x","100ms_4x"]
    # # extens = ["1000ms_2x","1000ms_4x"]
    # fileNames = [path+"spr45_realJ_10m_det_"+exten for exten in extens]

    """ STEP 10 m power pulses PI """
    # path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # # extens = ["1ms_2x","1ms_4x","10ms_2x","10ms_4x","100ms_4x"]#,"1000ms_2x","1000ms_4x"]
    # # extens = ["1ms_2x","1ms_4x"]
    # # extens = ["10ms_2x","10ms_4x"]
    # # extens = ["100ms_4x"]
    # extens = ["1000ms_2x","1000ms_4x"]
    # fileNames = [path+"spr45_realJ_10m_det_PI_"+exten for exten in extens]

    """ Resolution checks """
    # path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # # extens = ["ny800_dymin0-05","ny400_dymin0-05","ny200_dymin0-05","ny400_dymin0-25","ny400_dymin0-01","ny100_dymin0-05","ny200_dymin0-05_MC"]
    # extens = ["ny3200_dymin0-05","ny1600_dymin0-05","ny800_dymin0-05","ny400_dymin0-05","ny200_dymin0-05","ny100_dymin0-05"]
    # # extens = ["ny400_dymin0-05","ny400_dymin0-25","ny400_dymin0-01"]
    # # extens = ["ny800_dymin0-25","ny800_dymin0-10","ny800_dymin0-05","ny800_dymin0-01"]
    # # extens = ["ny200_dymin0-05","ny200_dymin0-05_MC"]
    # # extens = ["ny400_dymin0-05_rtol-1e-8","ny400_dymin0-05_rtol-1e-6","ny400_dymin0-05_rtol-1e-4","ny400_dymin0-05_rtol-1e-2"]
    # # extens = ["ny3200_dymin0-05","ny1600_dymin0-01_rtol-1e-8","ny1600_dymin0-05","ny800_dymin0-05","ny600_dymin0-025_rtol-1e-5"]
    # fileNames = [path+"spr45_realJ_10m_det_"+exten for exten in extens]

    """ Log grid vs standard """
    # path = "./simulations/WP_X/"
    # extens = ["PI_powScan_2-000x","logGrid_2x_200ms"]
    # fileNames = [path+"spr45_realJ_10m_det_"+exten for exten in extens]

    """ Rebounds """
    # path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
    # # extens = ["8m","6m","4m","2m","0m"]
    # # extens = ["0m","2m","4m","5m","6m","7m","8m","9m"]
    # fileNames = [path+"spr45_realJ_10m_det_2x-rebound-"+exten for exten in extens]

""" Osc Comp with const pow """
# path = "./simulations/WP_X/" # "/home/mbk513/Downloads/" #
# extens = ["powOsc_1-0_1-5x","PI_powScan_1-250x"]
# extens = ["powOsc_1-0_2-0x","PI_powScan_1-490x"]
# fileNames = [path+"spr45_realJ_10m_det_"+exten for exten in extens]

""" High Res Runs """
path = "/home/mbk513/Downloads/" # "./simulations/WP_X/" #
extens = ["ny800_dymin0-05_rtol1e-6_kla-0-2_fimp-1-5_udens-4-65","ny1600_dymin0-05_rtol1e-6_kla-0-2_fimp-1-5_udens-4-9"]
fileNames = [path+"spr45_realJ_dz-1_"+exten for exten in extens]


filesDict = {}
for fileName in fileNames:
    print(fileName)
    filesDict[str(fileName)] = loadData(fileName)

try:
    print(labels)
except:
    labels = extens

""" wtime check """
if(1):
    for fileName,exten in zip(fileNames,extens):
        print("file Exten: %-26s, num steps: %i"%(exten,len(filesDict[fileName]["t_array"])))
    print("##########")
    colours = cm.rainbow(np.linspace(0,1,len(fileNames)+1))
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    for fileName,colour,exten,lab in zip(fileNames,colours,extens,labels):
        plt.figure(1)
        ax1.plot(filesDict[fileName]["t_array"], filesDict[fileName]["wtime"],color=colour,label=lab)
        ax1.axvline(x=np.max(filesDict[fileName]["t_array"]),color=colour,linestyle="--")
        plt.figure(2)
        totTime = np.cumsum(filesDict[fileName]["wtime"])
        plt.plot(filesDict[fileName]["t_array"],totTime,color=colour,label=lab)
        plt.axvline(x=np.max(filesDict[fileName]["t_array"]),color=colour,linestyle="--")
        timePerStep = np.sum(filesDict[fileName]["wtime"])/len(filesDict[fileName]["t_array"])
        timePerTime = np.sum(filesDict[fileName]["wtime"])/filesDict[fileName]["t_array"][-1]
        print("file Exten: %-26s, total seconds: %-6i, sim time per real time: %i"%(exten,np.sum(filesDict[fileName]["wtime"]),timePerTime))
    plt.figure(1)
    ax1.set_zorder(1)
    ax2.set_zorder(-1)
    ax1.legend(loc="best")
    # ax1.set_yscale("log")
    ax1.set_xlabel("Real Time (ms)")
    ax2.set_xlabel("Time Index")
    ax1.set_ylabel("Sim. Time (s)")
    plt.figure(2)
    plt.legend(loc="best")
    plt.xlabel("Real Time (ms)")
    plt.xlabel("Time Index")
    plt.ylabel("Sim. Time (s)")
    # ax1.set_ylabel("ncalls")
    plt.show()
    # sys.exit()

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

logPlot    = True
startFrame = 0
stopFrame  = np.max([len(filesDict[fileName]["Ne"][:,:]) for fileName in fileNames])

colours = cm.rainbow(np.linspace(0,1,len(fileNames)+1))
if(len(fileNames)==2): colours = ["tab:red","tab:blue"]
if(len(fileNames)==3): colours = ["tab:red","tab:green","tab:blue"]

""" Density GIF """
if(0):
    animFigDens = plt.figure(2,figsize=(10,8))
    linesPlotted = []
    for fileName,colour,exten,lab in zip(fileNames,colours,extens,labels):
        linesPlotted.append(plt.plot([],color=colour,linestyle="--",label=lab,linewidth=2)[0])

    plt.xlim(min(filesDict[fileNames[0]]["cellPos"]),1.01*max(filesDict[fileNames[0]]["cellPos"]))
    if(logPlot):
        yMax = 1.01*np.max([np.max(filesDict[fileName]["Ne"][startFrame:stopFrame,:][filesDict[fileName]["Ne"][startFrame:stopFrame,:]>0.]) for fileName in fileNames])
        yMin = 0.99*np.min([np.min(filesDict[fileName]["Ne"][startFrame:stopFrame,:][filesDict[fileName]["Ne"][startFrame:stopFrame,:]>0.]) for fileName in fileNames])
    else:
        yMax, yMin = np.max([np.max(filesDict[fileName]["Ne"][startFrame:stopFrame,:]) for fileName in fileNames]), 0.0
    plt.ylim(yMin,yMax)
    plt.xlabel("cellPos (m)")
    plt.ylabel("Ne, (10^ m^-3)")
    if(logPlot): plt.yscale("log")
    from matplotlib.backend_bases import MouseButton
    Nframes = int(np.max([np.shape(filesDict[fileName]["Ne"][startFrame:stopFrame,:])[0] for fileName in fileNames]))#int(np.min([np.shape(filesDict[fileName]["Ne"][startFrame:,:])[0] for fileName in fileNames]))
    pause   = False
    frame   = 0
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
    def AnimationFunctionDens(frame):
        frame = control()
        plt.title("tind = %i"%(frame+startFrame))
        # if(logPlot):
        #     for linePlot,fileName in zip(linesPlotted,fileNames):
        #         if(frame>=len(filesDict[fileName]["Ne"][startFrame:stopFrame,:])):
        #             linePlot.set_data(([],[]))
        #         else:
        #             # linePlot.set_data((filesDict[fileName]["cellPos"], np.log10(filesDict[fileName]["Ne"][startFrame:,:][frame,:])-np.log10(filesDict[fileNames[-1]]["Ne"][startFrame:,:][frame,:])))
        #             linePlot.set_data((filesDict[fileName]["cellPos"], np.log10(filesDict[fileName]["Ne"][startFrame:stopFrame,:][frame,:])))
        # else:
        for linePlot,fileName in zip(linesPlotted,fileNames):
            if(frame>=len(filesDict[fileName]["Ne"][startFrame:stopFrame,:])):
                linePlot.set_data(([],[]))
            else:
                linePlot.set_data((filesDict[fileName]["cellPos"], filesDict[fileName]["Ne"][startFrame:stopFrame,:][frame,:]))
    animFigDens.canvas.mpl_connect("key_press_event", onPress)
    animFigDens.canvas.mpl_connect("button_press_event", onClick)
    anim_created = FuncAnimation(animFigDens, AnimationFunctionDens, frames=int(np.min([np.shape(filesDict[fileName]["Ne"][startFrame:stopFrame,:])[0] for fileName in fileNames])), interval=100)

    # plt.legend(loc="best",prop={"size":10})
    # f = r"./animation.gif"
    # writergif = PillowWriter(fps=10)
    # anim_created.save(f, writer=writergif)

    # anim_created.save("test.gif")
    plt.legend(loc="upper left",prop={"size":10})
    plt.show()
    # anim_created.save("test.gif")

rSquareds = []
# Final Profiles and evolutions
if(1):
    t0 = 0.
    tIndex = -1
    tStart = 0
    tStop  = -1
    popts  = []
    for i,(fileName,colour,lab) in enumerate(zip(fileNames,colours,labels)):
        print("#-----#")
        print(lab)
        #----- Detachment front position calculation
        cellPos, xPoint = filesDict[fileName]["cellPos"], filesDict[fileName]["xPoint"]
        cellPoshR = np.arange(np.min(cellPos),np.max(cellPos),0.001)
        detLoc = []
        for i in range(len(filesDict[fileName]["t_array"])):
            tempHR = np.interp(cellPoshR,cellPos,(filesDict[fileName]["Te"][i,:]+filesDict[fileName]["Td+"][i,:])/2.0)
            detPos = cellPoshR[-1]-cellPoshR[np.argmin(np.abs(tempHR-7.0))]
            detLoc.append(detPos)
        peakRad = cellPos[-1]-cellPos[np.argmax(filesDict[fileName]["Rimp"],axis=1)]

        detLoc    = np.copy(detLoc)[tStart:tStop]                              # Detachment Front in [m]
        rawTime   = 1e-3*(np.copy(filesDict[fileName]["t_array"][tStart:tStop])-t0)      # Time in [s]

        # plt.figure(1)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Ne"][tStart:tStop,-1], color=colour,linestyle="-", zorder=3,label=lab)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Ne"][tStart:tStop,0],  color=colour,linestyle=":", zorder=2)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Nd+"][tStart:tStop,-1],color=colour,linestyle="-", zorder=3)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Nd+"][tStart:tStop,0], color=colour,linestyle=":", zorder=2)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Nd"][tStart:tStop,-1], color=colour,linestyle="-", zorder=3)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Nd"][tStart:tStop,0],  color=colour,linestyle=":", zorder=2)
        # if(tIndex!=-1): plt.axvline(x=filesDict[fileName]["t_array"][tIndex],color="tab:grey",linestyle="--")
        # ##################
        # plt.figure(2)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,np.sum(filesDict[fileName]["Pe"][tStart:tStop,1:-1]*filesDict[fileName]["cellVolume"][1:-1],axis=1),color=colour,  linestyle="-", zorder=3,label=lab)
        # if(tIndex!=-1): plt.axvline(x=filesDict[fileName]["t_array"][tIndex],color="tab:grey",linestyle="--")
        # ##################
        # plt.figure(3)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,np.sum(filesDict[fileName]["Nd+"][tStart:tStop,1:-1]*filesDict[fileName]["cellVolume"][1:-1],axis=1),color=colour,linestyle="-",zorder=3,label=lab)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,np.sum(filesDict[fileName]["Nd"][tStart:tStop,1:-1]*filesDict[fileName]["cellVolume"][1:-1],axis=1),color=colour,linestyle="--",zorder=3)
        # totalParticleNumber = np.sum(filesDict[fileName]["Nd+"][tStart:tStop,1:-1]*filesDict[fileName]["cellVolume"][1:-1],axis=1)+np.sum(filesDict[fileName]["Nd"][tStart:tStop,1:-1]*filesDict[fileName]["cellVolume"][1:-1],axis=1)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,totalParticleNumber, color=colour, linestyle=":",zorder=3)
        # if(tIndex!=-1): plt.axvline(x=filesDict[fileName]["t_array"][tIndex],color="tab:grey",linestyle="--")
        # ##################
        # plt.figure(4)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Te"][tStart:tStop,-1], color=colour,linestyle="-", zorder=3,label=lab)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,filesDict[fileName]["Te"][tStart:tStop,0],  color=colour,linestyle=":", zorder=2)
        # if(tIndex!=-1): plt.axvline(x=filesDict[fileName]["t_array"][tIndex],color="tab:grey",linestyle="--")
        # ##################
        # plt.figure(5)
        # plt.plot(cellPos[:],filesDict[fileName]["Ne"][tIndex,:],linestyle="-", marker=".",color=colour,label=lab, zorder=2)
        # plt.plot(cellPos[:],filesDict[fileName]["Nd"][tIndex,:],linestyle="--",marker="", color=colour,zorder=1)
        # print("Upstream density = %.3e m^-3"%filesDict[fileName]["Ne"][tIndex,0])
        # ##################
        # plt.figure(6)
        # plt.plot(cellPos[:],filesDict[fileName]["Te"][tIndex,:],color=colour,zorder=2,label=lab)
        # # plt.plot(cellPos[:],np.gradient(filesDict[fileName]["Te"][tIndex,:],cellPos[:]),color=colour,zorder=2,label=lab)
        # ##################
        # plt.figure(7)
        # plt.plot(cellPos[:],filesDict[fileName]["Pe"][tIndex,:],color=colour,zorder=2,label=lab)
        # plt.plot(cellPos[:],filesDict[fileName]["Pd"][tIndex,:],color=colour,zorder=2)
        # # print("Target Pressure  = %.3e Pa"%filesDict[fileName]["Pd"][tIndex,-1])
        ##################
        plt.figure(8)
        plt.plot(rawTime,1e2*detLoc, linestyle="-", color=colour,label=lab)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,1e2*np.array(detLoc[tStart:tStop]), linestyle="-", color=colour,label=lab)
        # plt.plot(filesDict[fileName]["t_array"][tStart:tStop]-t0,1e2*np.array(peakRad[tStart:tStop]),linestyle="--",color=colour)#,label=lab)
        print("Front position   = %.3e m"%detLoc[tIndex])
        ##################
        plt.figure(9)
        xTime     = np.copy(rawTime)[(rawTime>=0.0)*(detLoc>0.5)][:np.argmin(detLoc)]#
        detLocFit = np.copy(detLoc)[(rawTime>=0.0)*(detLoc>0.5)][:np.argmin(detLoc)]#
        if(isinstance(lab,float)):
            guess = [10.*(lab-1.0)*0.772]#,np.min(detLoc)]
        else:
            guess = [5.]
        def expo(t,k):
            return np.max(detLoc)*np.exp(-k*t)
        popt, pcov = curve_fit(expo,xTime,detLocFit,p0=guess)
        if(i==0):
            popts = popt
        else:
            popts = np.append(popts,popt,axis=0)
        from sklearn.metrics import r2_score
        # coefficient_of_dermination = r2_score(detLocFit, expo(xTime,*popt))
        # rSquared = 1.0-( np.sum((detLocFit-expo(xTime,*popt))**2.) / np.sum((detLocFit-np.mean(detLocFit))**2.) )
        # print(rSquared,coefficient_of_dermination)
        # print(rSquared)
        # rSquareds.append(rSquared)
        plt.plot(rawTime[:np.argmin(detLoc)],detLoc[:np.argmin(detLoc)],              linestyle="-", color=colour,label=lab)
        plt.plot(rawTime[np.argmin(detLoc):],detLoc[np.argmin(detLoc):],              linestyle="--", color=colour,label=lab)
        # plt.plot(rawTime,expo(rawTime,*guess),linestyle="--",color=colour)#,label="guess")
        plt.plot(rawTime,expo(rawTime,*popt), linestyle=":", color=colour)#,label="pop")
        ##################
        plt.figure(10)
        argBounce = np.where(detLoc==np.min(detLoc))[0][-1]
        xTime     = np.copy(rawTime)[argBounce:]
        minTime   = xTime[0]
        xTime    -= minTime
        detLocFit = np.copy(detLoc)[argBounce:]
        guess     = [9.877,0.3+np.min(detLocFit),6.]
        def expo(t,yMax,yMin,l):
            return yMax-(yMax-yMin)*np.exp(-l*t)
        lB, uB = [9.876,0.3+np.min(detLocFit)*0.999,0.], [9.878,0.3001+np.min(detLocFit)*1.001,np.inf]
        # lB, uB = [9.876,0.,0.], [9.878,np.inf,np.inf]
        # lB, uB = [0.0,np.min(detLocFit)*0.999,0.], [np.inf,0.0001+np.min(detLocFit)*1.001,np.inf]
        popt, pcov = curve_fit(expo,xTime,detLocFit,p0=guess,bounds=(lB,uB))
        plt.plot(rawTime[(detLoc<43.0)],detLoc[(detLoc<43.0)],linestyle="-",color=colour,label=lab)
        # plt.plot(xTime+minTime, expo(xTime,*guess),linestyle="--",color=colour,label="guess")
        plt.plot(xTime+minTime, expo(xTime,*popt), linestyle=":", color=colour,label="popt")
        print(guess,popt)

    # plt.figure(1)
    # plt.title("Plasma Density Evolution")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Ne, t_array[:] (1e20 m^-3)")
    # plt.tight_layout()
    # ##################
    # plt.figure(2)
    # plt.title("Summed Plasma Pressure Evolution")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("np.sum(P), t_array[:]")
    # plt.tight_layout()
    # ##################
    # plt.figure(3)
    # plt.title("Total Particle Number Evolution")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("np.sum(Ne)")
    # plt.tight_layout()
    # ##################
    # plt.figure(4)
    # plt.title("Plasma Temperature Evolution")
    # plt.legend(loc="best")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Te, t_array[:] (eV)")
    # plt.tight_layout()
    # ##################
    # plt.figure(5)
    # # plt.axvline(x=78.13-0.05,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.10,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.20,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.40,color="tab:grey",linestyle="--")
    # plt.title("Final Density Profiles")
    # plt.legend(loc="best")
    # plt.xlabel("Cell Posistion (m)")
    # plt.ylabel("Particle Density (m^-3)")
    # plt.tight_layout()
    # ##################
    # plt.figure(6)
    # # plt.axvline(x=78.13-0.05,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.10,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.20,color="tab:grey",linestyle="--")
    # # plt.axvline(x=78.13-0.40,color="tab:grey",linestyle="--")
    # plt.legend(loc="best")
    # plt.title("Final Temperature Profile")
    # plt.xlabel("Cell Posistion (m)")
    # plt.ylabel("Plasma Temperature (eV)")
    # plt.tight_layout()
    # ##################
    # plt.figure(7)
    # plt.legend(loc="best")
    # plt.title("Final Pressure Profile")
    # plt.xlabel("cellPos (m)")
    # plt.ylabel("P[:], t_array[-1]")
    # plt.tight_layout()
    ##################
    plt.figure(8)
    # plt.xlim([0,5])
    # plt.ylim([0,100])
    plt.title("Det. Point Movement")
    plt.legend(loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Detachment Front (cm)")
    plt.tight_layout()
    # ##################
    # plt.figure(9)
    # plt.title("Front Movement Fits")
    # plt.legend(loc="best")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Detachment Front (m)")
    # plt.tight_layout()
    ##################
    plt.figure(10)
    plt.title("Front Rebound Fits")
    plt.legend(loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Detachment Front (m)")
    plt.tight_layout()
    # # plt.show()

# Heat flux on target
if(0):
    fig = plt.figure()#figsize=(9,5))
    mpl.rcParams['font.size']=14
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    # fig.suptitle("Target Heat Flux")
    # colours = ["tab:red","tab:blue"]
    for i,fileName,colour,exten,lab in zip(range(len(fileNames)),fileNames,colours,extens,labels):
        bohmVel = np.sqrt(2.*1.6e-19*filesDict[fileName]["Te"]/(2.5*1.67e-27))
        maxVel  = [np.max([i,j]) for i,j in zip(filesDict[fileName]["Vi"][:,-1],bohmVel[:,-1])]
        filesDict[fileName]["Q_t"] = 7.0*filesDict[fileName]["Ne"][:,-1]*(1.6e-19)*filesDict[fileName]["Te"][:,-1]*maxVel

        # tPow = 5
        # inputPower = [(0.772e+09)*i for i in [1,1,np.sqrt(tPow),np.sqrt(tPow),tPow,tPow,1,1]]
        # inputTimes = [filesDict[fileName]["t_array"][i] for i in [0,1000,1000,1025,1025,1125,1125,2125]]

        # ax2.plot(np.arange(len(filesDict[fileName]["Q_t"])),1e-6*filesDict[fileName]["Q_t"],     color="white",   linewidth=2,linestyle="-", alpha=0)
        # ax1.plot((filesDict[fileName]["t_array"])-10.5,      1e-6*filesDict[fileName]["Q_t"],     color=colour,linewidth=2,linestyle="-", label=lab,zorder=10)
        ax1.plot((filesDict[fileName]["t_array"])-t0,      1e-6*filesDict[fileName]["Q_t"],     color=colour,linewidth=2,linestyle="-", label=lab,zorder=10)

        # ax2.plot(np.arange(len(filesDict[fileName]["Q_t"]))-1000.,1e-6*filesDict[fileName]["Q_t"],     color="white",   linewidth=2,linestyle="-", alpha=0)
        # ax1.plot((filesDict[fileName]["t_array"])-10.,        1e-6*filesDict[fileName]["Q_t"],     color=colour,linewidth=2,linestyle="-", label=lab,zorder=10)#,marker="x")
        # if(i==len(fileNames)-1):ax1.plot(np.array(inputTimes)-10.,             1e-9*np.array(inputPower),color="k",       linewidth=2,linestyle="--",label="Input Power")
        # ax1.plot((filesDict[fileName]["t_array"])-10.,        1e-3*filesDict[fileName]["Te"][:,-1],color="tab:red", linewidth=2,linestyle=":", label="Target Temp. (keV)")
    # ax1.axhline(y=20.,linestyle="--",color="tab:grey")
    # ax2.set_xlim([-250,750])
    ax1.set_xlim([0,5])
    # ax1.set_ylim([1e-2,4e0])
    # plt.ylim([6.5e5,1.15e10])
    ax1.set_yscale("log")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Target Heat Flux (MW/m$^2$)")
    ax1.legend(loc="best",prop={"size":8})
    # ax2.set_xlabel("Time index")
    ax1.set_zorder(1)
    # ax2.set_zorder(-1)
    plt.tight_layout()
    # plt.savefig("step1000_8x.png")
    plt.show()
    # sys.exit()

plt.show()
# sys.exit()

power0      = 0.772 # 7.72e8
poptsArr    = np.reshape(np.ravel(popts),(len(fileNames),len(popt)))
powerFactor = power0*np.array([float(ext.replace("-",".").replace("x","")) for ext in extens])
powerDiff   = powerFactor-power0
polyFitFac  = np.poly1d(np.polyfit(powerFactor, np.ravel(poptsArr[:,0]), 1))
polyFitDiff = np.poly1d(np.polyfit(powerDiff,   np.ravel(poptsArr[:,0]), 1))
print(polyFitFac)
print(polyFitDiff)

plt.figure()
plt.plot(labels,rSquareds,marker="x")
plt.ylabel("R$^2$")
plt.xlabel("Power Factor")
plt.figure()
plt.plot(powerDiff,rSquareds,marker="x")
plt.ylabel("R$^2$")
plt.xlabel(r"$P_{Diff}$")
plt.show()

sys.exit()

plt.figure()
plt.plot(powerFactor,poptsArr[:,0],          linestyle="-", marker="x",color="tab:blue", label="Param")
plt.plot(powerFactor,polyFitFac(powerFactor),linestyle="--",marker="+",color="tab:green",label="Linear Fit, y = %.1f*x + %.1f"%(polyFitFac[1],polyFitFac[0]))
plt.xlabel("Power Factor")
plt.ylabel("k")
plt.legend(loc="best")
plt.figure()
plt.plot(powerDiff,poptsArr[:,0],         linestyle="-", marker="x",color="tab:blue", label="Fitted k")
plt.plot(powerDiff,polyFitDiff(powerDiff),linestyle="--",marker="+",color="tab:green",label="Linear Fit, k = %.1f*P[GW] + %.1f"%(polyFitDiff[1],polyFitDiff[0]))
plt.xlabel(r"$P_{Diff}$")
plt.ylabel("k")
plt.legend(loc="best")
# plt.figure()
# plt.plot(powers,popts[:,1],marker="x",color="tab:green")
# plt.xlabel("Power Factor")
# plt.ylabel("y0")
# plt.figure()
# plt.plot(powers,popts[:,2],color="tab:red")
# plt.xlabel("Power Factor")
# plt.ylabel("y0")
plt.show()
