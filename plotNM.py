import numpy as np
import sys,os
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

pd.set_option('display.max_rows',None)

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    # from whichcraft import which
    from shutil import which
    return which(name) is not None

import TOOLBOX as TLB
import CONSTANTS as CST

filename=sys.argv[1]
filetag=filename.split(".")[0]
molecule=filename.split("_")[0]
frequencies,reducedMasses,printedModesCart,atomicNumbers,atomicMasses,coordinates=TLB.fchk2vibrationalAnalysis(filename)
ESCF=TLB.fchk2derivatives(filename,mw=False,freq=True)[0]
NAtoms=len(atomicNumbers)
NCoords=len(coordinates.flatten())
NModes=NCoords-6
print("Number of atoms:",NAtoms)
print("Number of coordinates:",NCoords)
print("Number of normal modes of vibration",NModes)
print("ESCF: ",ESCF)

df=pd.DataFrame({"Freq (rcm)":frequencies,"Freq (a.u.)":(frequencies/CST.HARTREE_TO_RCM)**2,"μ (AMU)":reducedMasses,"μ (m_e)":reducedMasses*CST.AMU_TO_ME})
df.index=np.append(["TR"]*6,range(1,len(coordinates)+1-6))
print(df)

def collect_modes():
    arguments=sys.argv[2:]
    modes=[]
    for argument in arguments:
        # try:
            # mode=int(argument)
            # modes.append(mode)
        try:
            mode=str(argument)
            modes.append(mode)
        except ValueError:
            print(f"Ignored : {argument} is not a valid string.")
            # print(f"Ignored : {argument} is not a valid integer.")
    return modes

scale=2
if scale!=1:
    print("CAREFULL, scale for NM plot is not 1")
modeSelection=collect_modes()
grid=False
for mode in modeSelection:
    if "o" in mode:
        mode_index=int(mode.replace("o",""))-1+6
        fac=-1
    else:
        mode_index=int(mode)-1+6
        fac=1
    fig,ax=TLB.visualizeNormalMode(coordinates.flatten(),fac*printedModesCart[mode_index],atomicNumbers=atomicNumbers,projection="2d",scale=scale)
    ax.set_aspect("equal")
    # fig.suptitle("{} mode {}".format(filetag,mode))
    if grid:
        plt.grid()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        plt.axis("off")
    plt.tight_layout()
    pdfname="{}_mode_{}.pdf".format(filetag,mode)
    plt.savefig(pdfname)
    plt.show()
    if is_tool("pdfcrop"):
        os.system("pdfcrop {}".format(pdfname))

