import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

""" Logisitic function """
def dy_func(x,y_0,L,k,x_0):
    return (y_0+L/(1.+np.exp(-k*(x-x_0))))

""" Calculate the dy (and cell positions) based on logistic function """
def logisticGrid(yRange,y_0,L,k,x_0):
    dy_s    = []
    cellPos = []
    totalDist = 0.
    for i,y in enumerate(yRange):
        dy  = dy_func(y,y_0,L,k,x_0)
        dy -= 0.015*(np.exp(y-(77.855/78.13)*2.*np.pi))**2.
        totalDist += dy
        dy_s.append(dy)
        cellPos.append(totalDist)
    dy_s    = np.array(dy_s)
    cellPos = np.array(cellPos)
    """ Summing the dy's rather than finding their cell centres """
    dySum     = np.copy(dy_s)
    dySum[1:] = [np.sum(dySum[:i]) for i in range(2,len(dy_s)+1)]
    totalDist = dySum[-1]
    return dy_s,cellPos,totalDist#,cellPos_centre

""" Standard grid setup of increasing resolution towards the target """
def standardGrid(yRange,length,length_xpt,ny,dymin):
    dy     = (length/ny)*(1.+(1.-dymin)*(1.-yRange/np.pi))
    source = length_xpt / length
    y_xpt  = np.pi*(2.0-dymin-np.sqrt((2.0-dymin)**2.0-4.0*(1.0-dymin)*source))/(1.0-dymin)

    dySum     = np.copy(dy)
    dySum[1:] = [np.sum(dySum[:i]) for i in range(2,len(dy)+1)]
    cellPos   = dySum
    totalDist = dySum[-1]
    return dy,cellPos,totalDist#,cellPos_centre

""" Base features """
length     = 78.13
length_xpt = 35.14 # 57.98
areaExp    = 2.051 # 1.818

""" Standard Set up """
ny_stand    = 1600
dymin_stand = 0.050

""" det10m """
ny_log  = 512
L_log   = 0.9518 # 0.94248
k_log   = -4. # -1.5
x_0_log = (1./7.5)*2.*np.pi
y_0_log = 0.025

#----- Comparing different standard grids
if(0):
    for ny in [100,200,400,800,1600,3200]:
        yRange_stand = np.linspace(0,2.*np.pi,ny)
        dy_stand, cellPos_stand, totalDist_stand = standardGrid(yRange_stand,length,length_xpt,ny,dymin_stand)
        # plt.plot(cellPos_stand,1e2*dy_stand,marker=".",zorder=0,label="ny=%i, dymin=%.3f"%(ny,dymin_stand))
        plt.plot(cellPos_stand,1e2*dy_stand,zorder=0,label="ny=%i, dymin=%.3f"%(ny,dymin_stand))
    plt.ylabel("dy (cm)")
    plt.xlabel("y (m)")
    plt.ylim(bottom=0.0)
    plt.axvline(x=length_xpt,color="tab:grey",linestyle="--")
    plt.legend(loc="best")
    plt.show()
    for dymin,linestyle in zip([0.01,0.05,0.25],["-","--",":"]):
        yRange_stand = np.linspace(0,2.*np.pi,400)
        dy_stand, cellPos_stand, totalDist_stand = standardGrid(yRange_stand,length,length_xpt,400,dymin)
        # plt.plot(cellPos_stand,1e2*dy_stand,marker=".",zorder=0,label="ny=%i, dymin=%.3f"%(400,dymin))
        plt.plot(cellPos_stand,1e2*dy_stand,linestyle=linestyle,linewidth=2,zorder=0,label="ny=%i, dymin=%.3f"%(400,dymin))
    plt.ylabel("dy (cm)")
    plt.xlabel("y (m)")
    plt.ylim(bottom=0.0)
    plt.axvline(x=length_xpt,color="tab:grey",linestyle="--")
    plt.legend(loc="best")
    plt.show()
    sys.exit()

#----- Standard
yRange_stand = np.linspace(0,2.*np.pi,ny_stand)
dy_stand, cellPos_stand, totalDist_stand = standardGrid(yRange_stand,length,length_xpt,ny_stand,dymin_stand)
#----- Logistic
yRange_log = np.linspace(0,2.*np.pi,ny_log)
dy_log, cellPos_log, totalDist_log = logisticGrid(yRange_log,y_0_log,L_log,k_log,x_0_log)

print("Standard length = %.3f, goal of %.2f"%(totalDist_stand,length))
print("Logisitc length = %.3f, goal of %.2f"%(totalDist_log,length))

dy, cellPos, totalDist, yRange = dy_log,   cellPos_log,   totalDist_log,   yRange_log
# dy, cellPos, totalDist, yRange = dy_stand, cellPos_stand, totalDist_stand, yRange_stand

#----- Compare dy
plt.figure()
plt.plot(cellPos_log,  1e2*dy_log,  marker="x",zorder=1,label="ny=%i, y0=%.3f"%(ny_log,y_0_log))
plt.plot(cellPos_stand,1e2*dy_stand,marker=".",zorder=0,label="ny=%i, dymin=%.3f"%(ny_stand,dymin_stand))
plt.ylabel("dy (cm)")
plt.xlabel("y (m)")
plt.ylim(bottom=0.0)
plt.axvline(x=length_xpt,color="tab:grey",linestyle="--")
plt.legend(loc="best")
# plt.show()

xPointArg  = np.argmin(np.abs(cellPos-length_xpt))
calc_y_xpt = yRange[xPointArg]

print("\nny = %i"%ny_log)
print("length_xpt = %.1f"%length_xpt)
print("y_0 = %.5e"%y_0_log)
print("L = %.5e"%L_log)
print("k = %.5e"%k_log)
print("x_0 = %.5e"%x_0_log)
print("dy = (y_0+L/(1.0+exp(-k*(y-x_0))))-0.015*(exp(y-(77.855/78.13)*2.*pi))**2.")
print("y_xpt = %.5e\n"%calc_y_xpt)

xPointArg  = np.argmin(np.abs(cellPos-length_xpt))
calc_y_xpt = yRange[xPointArg]

""" Load in B-profile """
bprofDF = pd.read_csv("~/Documents/Projects/STEP_lot2/fieldLineCalcs/outer_b-field_profiles.csv")
J_goal  = bprofDF["B_tot"].values
""" Interp points to premade grid """
J_goal  = np.interp(cellPos,bprofDF["length"].values,J_goal)
""" Convert to flux expansion """
J_goal  = J_goal[xPointArg]/J_goal
""" apply weighting to first and last points (x-point and target) """
weights            = np.ones(len(cellPos))
weights[0]         = 1.
weights[xPointArg] = 1.
weights[-1]        = 1.
""" Perform the polyfit """
pfP         = np.polyfit(yRange,J_goal,deg=10,w=weights)
polyfitFunc = np.poly1d(pfP)
printStatement = "J = 0 "
for i,coef in enumerate(pfP[::-1]):
    print("coef%i = %.10e"%(i,coef))
    printStatement += " + "+"y^"+str(i)+"*coef%i"%(i)
print(printStatement)
xpt_vol = np.sum((dy*polyfitFunc(yRange))[:xPointArg])
print("xpt_vol = %.5f"%xpt_vol)
# print("\narea = 1 + ((%.5e*y*y*y*y*y*y)+(%.5e*y*y*y*y*y)+(%.5e*y*y*y*y)+(%.5e*y*y*y)+(%.5e*y*y)+(%.5e*y)+%.5e)"%(pfP[0],pfP[1],pfP[2],pfP[3],pfP[4],pfP[5],pfP[6]))
plt.figure()
plt.subplot(2,1,1)
plt.plot(cellPos,J_goal,             label="Goal J")
plt.plot(cellPos,polyfitFunc(yRange),label="Fitted J",linestyle="--")
plt.axvline(x=length_xpt,color="tab:grey",linestyle="--")
plt.legend(loc="best")
plt.xlabel("cellPos (m)")
plt.ylabel("J")
# plt.ylim([0.9,1.1*areaExp])

plt.subplot(2,1,2)
plt.plot(yRange,J_goal,             label="Goal J")
plt.plot(yRange,polyfitFunc(yRange),label="Fitted J",linestyle="--")
plt.axvline(x=calc_y_xpt,color="tab:grey",linestyle="--")
plt.legend(loc="best")
plt.xlabel("yRange (index space)")
plt.ylabel("J")
# plt.ylim([0.9,1.1*areaExp])
plt.tight_layout()

print("\nVolume up to X-point = %.3f"%(np.sum((dy*polyfitFunc(yRange))[:xPointArg])))
print("length_xpt = ",length_xpt)
plt.show()
