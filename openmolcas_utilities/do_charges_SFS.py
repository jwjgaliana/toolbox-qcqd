#!/usr/bin/env python3
import h5py,tarfile
import sys,os
import numpy as np
import pandas as pd
import numpy.linalg
import subprocess

cwd_start=os.getcwd()
################################
##  Parameters                ##
################################
snapshots=np.arange(0,101,1) # all steps should have the required files
name="full-all-SFS"
analysis_dir="chargeAnalysis_{}".format(name)
if not os.path.isdir(analysis_dir):
    os.makedirs(analysis_dir)
os.chdir(analysis_dir)

filename_charges="charges.tsh-density-{}.dat".format(name)
file_charges=open(filename_charges,"w")
do_cubes=False

def do_svd(matrix,lapack_driver="gesvd",full_matrices=True,debug_print=False,result_print=True):
    U,s,Vh=scipy.linalg.svd(matrix,full_matrices=full_matrices,lapack_driver=lapack_driver)
    if debug_print:
        print("U.shape", U.shape)
        print("s.shape", s.shape)
        print("s",s)
        print("Vh.shape", Vh.shape)
    ExcitationAmplitudes=np.copy(s) # s, or capital Lambda
    MEigenvalues=ExcitationAmplitudes**2 # lambda, eigenvalues of DDdagger
    ExcitationContributions=MEigenvalues/np.sum(MEigenvalues)
    if result_print:
        pd.options.display.float_format = '{:,.5f}'.format
        df=pd.DataFrame({"ExcitationAmplitudes":ExcitationAmplitudes,"MEigenvalues":MEigenvalues,"ExcitationContributions":ExcitationContributions},index=np.arange(1,len(ExcitationAmplitudes)+1))
        print(df.head().to_string())
    HoleNOVectors=np.copy(U)
    PartNOVectors=np.copy(Vh.T)
    if debug_print:
        print("UUT",np.dot(U,U.T))
        print("VVT",np.dot(Vh,Vh.T))
        print("reconstructed matrix",np.dot(U,np.dot(np.diag(s),Vh))) # U s VT
        print("reconstructed S",np.dot(U.T,np.dot(matrix,Vh.T))) # UT D V
    return ExcitationAmplitudes,ExcitationContributions,HoleNOVectors,PartNOVectors

def read_cube(filename):
    with open(filename,"r") as f:
        lines=f.readlines()
        title=lines[0]
        NGridPoints=int(lines[1].split()[1])
        NAtoms=int(lines[2].split()[0])
        startCoordinates=np.array(lines[2].split()[1:],dtype=float)
        Nx,dx=int(lines[3].split()[0]),float(lines[3].split()[1])
        Ny,dy=int(lines[4].split()[0]),float(lines[4].split()[2])
        Nz,dz=int(lines[5].split()[0]),float(lines[5].split()[3])
        atomsInformation=np.array([line.split() for line in lines[6:6+NAtoms]],dtype=object)
        atomsCoordinates=atomsInformation[:,2:].astype(float)
        atomicNumbers=atomsInformation[:,0].astype(float)
        cubeData=np.ones((Nx,Ny,Nz),type(float))
        if Nz % 6 == 0:
            nlines = Nz // 6
        else:
            nlines = Nz // 6 + 1
        line = NAtoms + 6
        for x in range(Nx):
            for y in range(Ny):
                z = 0
                for j in range(nlines):
                    line_data = lines[line].split()
                    for i in range(len(line_data)):
                        cubeData[x,y,z] = float(line_data[i])
                        z += 1
                    line += 1
    return cubeData

def write_cube(cubeData,filename,header=None):
    print("  writing {}".format(filename))
    Nx,Ny,Nz=cubeData.shape
    if Nz % 6 == 0:
        nlines = Nz // 6
    else:
        nlines = Nz // 6 + 1
    # print("nlines",nlines)
    line=0
    with open(filename,"w") as f:
        if header is not None:
            f.write("".join(header))
        for x in range(Nx):
            for y in range(Ny):
                z = 0
                for j in range(nlines-1):
                    for i in range(6):
                        f.write(str(cubeData[x,y,z])+"\t")
                        z += 1
                    f.write("\n")
                    line += 1
                ## does work if ... last "line" of z is // by 6
                # for i in range(Nz % 6):
                    # f.write(str(cubeData[x,y,z])+"\t")
                    # z += 1
                ## works even if ... last "line" of z is // by 6
                # if Nz % 6 != 0:
                    # for i in range(Nz % 6):
                        # f.write(str(cubeData[x,y,z])+"\t")
                        # z += 1
                # elif Nz % 6 == 0:
                    # for i in range(6):
                        # f.write(str(cubeData[x,y,z])+"\t")
                        # z += 1
                ## should work any case... check length of last "line" of z
                for i in range(6*((Nz-z) // 6) + Nz % 6):
                    f.write(str(cubeData[x,y,z])+"\t")
                    z += 1
                f.write("\n")
                # print("z",z)
    return


cwd = os.getcwd()

#################################################
## Get coefficients                            ##
#################################################
print("#################################################")
print("## Get coefficients                            ##")
print("#################################################")

data=np.loadtxt("../coef_raw.out",dtype=float)
times=data[:,0]
coefficients=data[:,2:]
realParts=coefficients[:,::2]
imagParts=coefficients[:,1::2]
amplitudes=realParts**2+imagParts**2
NStates=len(amplitudes[0])//4
print("coefficients shape (raw)",coefficients.shape)
print("select only singlets and store as complex numbers")
coefficients=realParts+1j*imagParts
all_coefficients=np.copy(coefficients)
# coefficients          singlets
# coefficients_msm1     triplets(ms=-1)
# coefficients_ms0      triplets(ms= 0)
# coefficients_msp1     triplets(ms=+1)
coefficients=all_coefficients[:,:NStates]
coefficients_msm1=all_coefficients[:,NStates+0::3]
coefficients_ms0=all_coefficients[:,NStates+1::3]
coefficients_msp1=all_coefficients[:,NStates+2::3]
dico_coefficients={}
dico_coefficients["SINGLETS"]=coefficients
dico_coefficients["TRIPLETS(msm1)"]=coefficients_msm1
dico_coefficients["TRIPLETS(ms0)"]=coefficients_ms0
dico_coefficients["TRIPLETS(msp1)"]=coefficients_msp1
# realParts=coefficients.real
# imagParts=coefficients.imag
# amplitudes=realParts**2+imagParts**2
print("coefficients shape (only singlets)",coefficients.shape)
print("coefficient in time are\n",pd.DataFrame(coefficients))
print("coefficients shape (only triplets(-1))",coefficients_msm1.shape)
print("coefficient in time are\n",pd.DataFrame(coefficients_msm1))
print("coefficients shape (only triplets( 0))",coefficients_ms0.shape)
print("coefficient in time are\n",pd.DataFrame(coefficients_ms0))
print("coefficients shape (only triplets(+1))",coefficients_msp1.shape)
print("coefficient in time are\n",pd.DataFrame(coefficients_msp1))

#################################################
## Get transition density matrix (in AO basis) ##
#################################################
print("#################################################")
print("## Get transition density matrix (in AO basis) ##")
print("#################################################")

# snapshots=[0] # for testing
# snapshots=np.arange(0,101,5)
for sp in snapshots:
    spdir="sp_{:09d}".format(sp)
    if not os.path.isdir(spdir):
        os.makedirs(spdir)
    calcs=["cas1","cas3"]
    print("We are here:",os.getcwd())
    density_in_AO=0
    tar_of_sp=os.path.join("..","SAVES","{}.tar".format(spdir))
    print("tar_of_sp",tar_of_sp)
    with tarfile.open(tar_of_sp,'r') as tar:
        print("tar",tar)
        print("tar.getmembers()",tar.getmembers())
        with tarfile.open("r:gz",fileobj=tar.extractfile(tar.getmembers()[-1])) as tar_of_h5:
            print("tar_of_h5",tar_of_h5)
            print("tar_of_h5.getmembers()",tar_of_h5.getmembers())
            for calc in calcs:
                # filename_rasscf=os.path.join(spdir,calc+".rasscf.h5") # old
                # filename_rassi=os.path.join(spdir,calc+".rassi.h5")   # old
                filename_rasscf=calc+".rasscf.h5"
                filename_rassi=calc+".rassi.h5"
                print("Initial file name (RASSCF) ",filename_rasscf)
                print("Initial file name (RASSI) ",filename_rassi)
                print("####################################################")
                print("## Reading quantum chemistry data from {}         ".format(calc))
                print("####################################################")

                directory_name=cwd
                # output_dir="stateDensityCube_from_trd1_rassi_via_NO"
                # if not os.path.isdir(output_dir):
                    # os.makedirs(output_dir)
                h5file=tar_of_h5.extractfile(tar_of_h5.getmember(filename_rasscf))
                # with h5py.File(directory_name+"/"+filename_rasscf,"r") as f:
                with h5py.File(h5file,"r") as f:
                    MOVectors=f['MO_VECTORS'][:].astype(float)
                    MOTypeIndices=f['MO_TYPEINDICES'][:].astype(str)
                # with h5py.File(directory_name+"/"+filename_rassi,"r") as f:
                h5file=tar_of_h5.extractfile(tar_of_h5.getmember(filename_rassi))
                with h5py.File(h5file,"r") as f:
                    NStates=f.attrs['NSTATE']
                    NBast=f.attrs['NBAS'][0]
                    NAtoms=f.attrs['NATOMS_UNIQUE']
                    AtomicNumbers=f['CENTER_ATNUMS'][:].astype(int)
                    AtomicLabels=f['CENTER_LABELS'][:].astype(str)
                    Coordinates=f['CENTER_COORDINATES'][:].astype(float)
                    Primitives=f['PRIMITIVES'][:].astype(float)
                    PrimitiveIDs=f['PRIMITIVE_IDS'][:].astype(int)
                    TDM=f['SFS_TRANSITION_DENSITIES'][:] # if RASSI file, in the basis of AO (or MO?)
                    TSDM=f['SFS_TRANSITION_SPIN_DENSITIES'][:] # if RASSI file, in the basis of AO (or MO?)
                    AOOverlap=f['AO_OVERLAP_MATRIX'][:] # if RASSI file, overlap matrix for the atomic orbitals
                hotfix=False
                if NStates==20:
                    hotfix=True
                if hotfix:
                    NStates=NStates//2
                    TDM=TDM[NStates:,NStates:,:]
                    TSDM=TSDM[NStates:,NStates:,:]
                pairs=np.array([(i,j) for i in range(NStates) for j in range(i,NStates)])
                NPairs=len(pairs)

                print("NBast: ",NBast)
                print("MOVectors.shape: ",MOVectors.shape)
                MOVectors=MOVectors.reshape((NBast,NBast))
                print("MOVectors new shape: ",MOVectors.shape) # MO vectors in rows?
                print("AOOverlap shape: ",AOOverlap.shape)
                AOOverlap=AOOverlap.reshape((NBast,NBast))
                print("AOOverlap new shape: ",AOOverlap.shape)
                CVectors=MOVectors.T
                print("CVectors.shape: ",CVectors.shape)

                # print(stop)

                print("C.T·S·C=(should be identity)\n",pd.DataFrame(np.dot(CVectors.T,np.dot(AOOverlap,CVectors))))

                print("NStates: ",NStates)
                print("NPairs with diagonal ",len(pairs))
                # print("Pairs: \n",pairs)
                print("TDM shape",TDM.shape)
                TDM=TDM.reshape((NStates,NStates,NBast,NBast))
                print("TDM new shape",TDM.shape)
                print("TSDM shape",TSDM.shape)
                TSDM=TSDM.reshape((NStates,NStates,NBast,NBast))
                print("TSDM new shape",TSDM.shape)
                TDM=0.5*(TDM+TSDM)
                print("TDM = 0.5*(TDM + TSDM) (transition spin density added)")

                # if calc=="cas1":
                if "cas1" in calc:
                    coefficients_sp_list=[dico_coefficients["SINGLETS"][sp]]
                # elif calc=="cas3":
                elif "cas3" in calc:
                    coefficients_sp_list=[
                            dico_coefficients["TRIPLETS(msm1)"][sp],
                            dico_coefficients["TRIPLETS(ms0)"][sp],
                            dico_coefficients["TRIPLETS(msp1)"][sp],
                            ]
                mscounter=0
                for coefficients_sp in coefficients_sp_list:
                    print("######################################")
                    print("##Expecting for {} with set {}      ".format(calc,mscounter))
                    print("######################################")
                    mscounter+=1
                    print("coefficients_sp.shape",coefficients_sp.shape)
                    print("coefficients_sp",coefficients_sp)
                    # density_in_AO=np.linalg.multi_dot([coefficients_sp.conjugate(),TDM,coefficients_sp])
                    density_in_AO+=np.dot(np.dot(coefficients_sp.conjugate(),TDM.T),coefficients_sp).T
                    # TODO check manually with two loops that this (.T) is indeed what we want to do
    print("(total) density_in_AO.shape",density_in_AO.shape)

    print("(total) density_in_AO\n",pd.DataFrame(density_in_AO))
    print("real part of density_in_AO\n",pd.DataFrame(density_in_AO.real))
    print("imag part of density_in_AO\n",pd.DataFrame(density_in_AO.imag))
    print("check that imag part is zero (should be)")
    density_in_AO=density_in_AO.real

    ####################################################
    ## Diagonalize density matrix with AOOverlap      ##
    ####################################################
    print("####################################################")
    print("## Diagonalize density matrix with AOOverlap      ##")
    print("####################################################")

    density_in_ortho=numpy.linalg.multi_dot([CVectors.T,AOOverlap,density_in_AO,AOOverlap,CVectors])
    print("density_in_AO transformed into density_in_ortho: ",density_in_ortho.shape)
    print("diagonalizing")
    eigenvalues,diagonalizer=np.linalg.eigh(density_in_ortho)
    NOOccupations=np.copy(eigenvalues)
    NOOccupations*=2
    NOVectors=np.copy(diagonalizer)
    NOVectors=numpy.linalg.multi_dot([CVectors,NOVectors])
    print("And transforming result NOVectors with C·NOVectors")
    print("NOOccupations\n",pd.DataFrame(NOOccupations))

    NOOccupations=np.array(NOOccupations,dtype=float)
    NOIndices=np.array(np.arange(len(NOOccupations))+1,dtype=int)
    NOOccupations=NOOccupations[::-1]
    NOVectors=NOVectors.T[::-1].T
    print("Re-order NOOccupations (descending)")
    print(pd.DataFrame(NOOccupations))
    print("NOOccupations\n",pd.DataFrame(np.array([NOOccupations,NOIndices]).T))
    # print(NOIndices)

    ################################################################
    ## Printing Natural Orbitals   as molden files                ##
    ################################################################

    occupations=np.copy(NOOccupations)
    filename_molden="NOs.tsh-density-{}.from-trd1-rassi.molden".format(name)
    with open(os.path.join(spdir,filename_molden),"w") as o:
        pd.options.display.float_format = '{:,.8f}'.format
        o.write("[Molden Format]\n")
        o.write("[N_Atoms]\n\t{}\n".format(NAtoms))
        o.write("[Atoms] (AU)\n")
        df=pd.DataFrame({"atomic center":np.arange(1,NAtoms+1),"atomic number":AtomicNumbers,"x":Coordinates[:,0],"y":Coordinates[:,1],"z":Coordinates[:,2]},index=AtomicLabels)
        o.write(df.to_string(header=False)+"\n")
        o.write("[5D]\n")
        o.write("[7F]\n")
        o.write("[9G]\n")
        o.write("[GTO] (AU)\n")
        pd.options.display.float_format = '{:,.9E}'.format
        # TODO write the primitives looping over the (angmom,shell_types) possibilities...
        Primitives_dic={}
        PrimitivesPerAtom=[]
        OrbitalsPerAtom=[]
        ShellLabels=["s","p","d","f","g"]
        for _ in range(1,NAtoms+1):
            nprimitives=0
            norbitals=0
            where=(PrimitiveIDs[:,0]==_)
            Primitives_dic[str(_)]=(PrimitiveIDs[where][:,1:],Primitives[where])
            o.write("   {}\n".format(_))
            dico=Primitives_dic[str(_)]
            nmax=np.max(dico[0][dico[0][:,0]==0])
            lmax=np.max(dico[0][:,0])
            # print("nmax is ",nmax)
            # print("lmax is ",lmax)
            # DONE write the primitives looping over the (angmom,shell_types) possibilities...
            # TODO check if this loop is correct for all types of atoms...
            for l in range(lmax+1):
                for n in range(1,nmax-l+1):
                    # nl
                    where=np.logical_and(dico[0][:,0]==l,dico[0][:,1]==n)
                    if len(where[where])>0:
                        nprimitives+=len(where[where])
                        norbitals+=2*l+1
                        o.write("   {}\t{}".format(ShellLabels[l],len(where[where]))+"\n")
                        df=pd.DataFrame({"tab":[""]*len(where[where]),"exp":dico[1][where][:,0],"cont":dico[1][where][:,1]})
                        o.write(df.to_string(header=False,index=False)+"\n")
            o.write("\n")
            PrimitivesPerAtom.append(nprimitives)
            OrbitalsPerAtom.append(norbitals)
            # print(f"Here, norbitals for {AtomicLabels[_-1]}\n",norbitals)
        # print("HERE",np.sum(OrbitalsPerAtom))
        pd.options.display.float_format = '{:,.8f}'.format
        o.write("[MO]\n")
        permutations=np.arange(NBast).astype(int)
        counter=0
        for i in range(NAtoms):
            dico=Primitives_dic[str(i+1)]
            nmax=np.max(dico[0][dico[0][:,0]==0])
            lmax=np.max(dico[0][:,0])
            # print("label, nmax, lmax",AtomicLabels[i],nmax,lmax)
            shell_types=ShellLabels[:lmax+1]
            permutation=[]
            d_permutation=[2,3,1,4,0]
            index=0
            for shell_type in shell_types:
                if shell_type=="s":
                    for n in range(nmax):
                        permutation.append(n)
                elif shell_type=="p":
                    for n in range(1,nmax):
                        for p in range(3):
                            permutation.append(nmax+(nmax-1)*p+(n-1))
                elif shell_type=="d":
                    for d in range(5):
                        permutation.append(nmax+(nmax-1)*2+(nmax-1)+d_permutation[d])
                # print(permutation)
            # print(f"Here, permutation for atom {AtomicLabels[i]} is ",permutation)
            # print(OrbitalsPerAtom[i])
            # print(len(permutation))
            permutations[np.arange(counter,counter+OrbitalsPerAtom[i])]=counter+np.array(permutation)
                # ns,npx,npy,npz,nd(-.+)
                # ns,n(px,py,pz),nd(0+-+-)
            counter+=OrbitalsPerAtom[i]
        for _,occ in enumerate(occupations):
            newMOVector=NOVectors.T[_][permutations]
            o.write("Sym= {}a\n".format(_+1))
            o.write("Ene= {}\n".format(0.0))
            o.write("Spin= Alpha\n")
            o.write("Occup= {}\n".format(occ))
            df=pd.DataFrame({"MOVector":newMOVector},index=np.arange(1,NBast+1))
            o.write(df.to_string(header=False)+"\n")

    #################################################
    ## Run Multiwfn to have ELF basins             ##
    #################################################
    print("#################################################")
    print("## Run Multiwfn to have ELF basins             ##")
    print("#################################################")

    # multiwfn="Multiwfn"
    multiwfn="/home/uam/uam382116/programs/Multiwfn_noGUI"
    arguments=[multiwfn,os.path.join(spdir,filename_molden)]
    # basins=["1","13","16","37"]
    basins=["a"]
    output_dir=os.path.join(spdir,filename_molden+".ALL_BASINS/")

    # use of subprocess might be overkill; can try to use bash only, look at p.1181 of Multiwfn manual
    process=subprocess.Popen(arguments,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    input_data=[
        ### BASIN ANALYSIS ###
        "17", # basins analysis
        "1", # generate basins...
        "9", # ... with ELF
        "1", # grid low quality (change to 2 later)
        ]
    if do_cubes:
        input_data+=[
            ### EXTRACT BASINS TO CUBE FILES ###
            "-5", # extract 
            ",".join(basins), # selected basins, basin0001 used for grid spec. in the following
            # "1", # only basins (0 or 1)
        ]
    input_data+=[
        ### ASSIGN ELF LABELS (in case useful) ###
        "12",
        ### INTEGRATE ELDENS WITHIN BASINS (in case useful) ###
        # "2", # integrate real space function
        # "1", # electron density
        "-10", # return to main menu
        ]
    if do_cubes:
        input_data+=[
            ### EXTRACT ELECTRON DENSITY TO CUBE FILE ###
            "5", # grid data
            "1", # electron density
            # "1", # grid low quality (change to 2 later)
            "8", # grid spec from other cub (next file)
            "basin.cub", #
            # "basin0001.cub", #
            "2", # save cube file
            "0", # return to main menu
            ]
    input_data+=[
        "q", # quit 
        ]
    input_data="\n".join(input_data)
    print(input_data)
    print("Will generate electron density and ELF...")
    print("Will generate basins based on ELF...")
    output,errors=process.communicate(input=input_data)
    with open(os.path.join(spdir,"log_multiwfn.tsh-density-{}".format(name)),"w") as f:
        f.write(output)
    if do_cubes:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        os.system("mv *.cub {}".format(output_dir))

    with open(os.path.join(spdir,"log_multiwfn.tsh-density-{}".format(name)),"r") as f:
        lines=f.readlines()
    dic_basins={}
    for iline,line in enumerate(lines):
        if "Sorting basins" in line:
            got_all_basins=False
            c=0
            while not got_all_basins:
                new_line=lines[iline+3+c]
                c+=1
                if "#" not in new_line:
                    got_all_basins=True
                else:
                    new_line=new_line.split()
                    label=new_line[-1]
                    label=label+"-a"
                    if label not in dic_basins.keys():
                        print(f"Label {label} not already in dictionary")
                        print(f"Label {label} stored as {label}")
                    # if label in dic_basins.keys(): # works only for one repetition
                    while label in dic_basins.keys(): # works for multiple repetitions
                        old_label=label
                        old_suffix=old_label.split("-")[-1]
                        label="-".join(old_label.split("-")[:-1]+[chr(ord(old_suffix)+1)])
                        print(f"Label {old_label} already in dictionary")
                        print(f"Label {old_label} modified to {label}")
                        # label=label+"p"
                    dic_basins[label]={}
                    dic_basins[label]["BASIN_INDEX"]=int(new_line[3])
                    dic_basins[label]["BASIN_POPULATION"]=float(new_line[5])
                    dic_basins[label]["BASIN_VOLUME"]=float(new_line[7])

    print(f"step {sp:09d} NBasins {len(dic_basins.keys()):04d}")
    print(pd.DataFrame(dic_basins).T)
    file_charges=open(filename_charges,"a+")
    file_charges.write(f"step {sp:09d} NBasins {len(dic_basins.keys()):04d}\n")
    file_charges.write(pd.DataFrame(dic_basins).T.to_string())
    file_charges.write("\n")
    file_charges.close()

    if do_cubes:
        with open(output_dir+"basin.cub","r") as f:
            lines=f.readlines()
        NAtoms=int(lines[2].split()[0])
        header=lines[:6+NAtoms]
        all_basins_cubeData=read_cube(output_dir+"basin.cub")

        density_cubeData=read_cube(output_dir+"density.cub")

        # valence_labels=["V(N3)","V(N8)","V(C1,N8)","V(N8,C1)","V(N3,C7)","V(C7,N3)"]
        valence_labels=["V(N8)","V(N11)","V(C2,N11)","V(N11,C2)","V(N8,C3)","V(C3,N8)"]
        for label in valence_labels:
            print(f"Valence {label} corresponds to...")
            for key in dic_basins.keys():
                if label in key:
                    basin_index=dic_basins[key]["BASIN_INDEX"]
                    basin_population=dic_basins[key]["BASIN_POPULATION"]
                    print("# Basin {} with charge {} with label {}".format(basin_index,basin_population,key))
                    basin_cubeData=np.zeros(all_basins_cubeData.shape)
                    # extract basin from basins.cub (value in basins.cub = basin index)
                    basin_cubeData[all_basins_cubeData==basin_index]=1.0
                    write_cube(basin_cubeData,output_dir+"basin{}.cub".format(key),header=header)
                    # multiply electron density with basin
                    eldens_basin_cubeData=basin_cubeData*density_cubeData
                    write_cube(eldens_basin_cubeData,output_dir+"eldens_basin{}.cub".format(key),header=header)

# old things
    # ",".join(["1","10","20","36"]), # selected basins, basin0001 used for grid spec. in the following

