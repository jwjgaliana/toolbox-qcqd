import numpy as np
import os,sys
import subprocess

def add_bool_arg(parser,name,default=False,doc="Bool type"):
    group=parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--'+name,dest=name,action='store_true',help=doc)
    group.add_argument('--no-'+name,dest=name,action='store_false',help=doc)
    parser.set_defaults(**{name:default})

def auto_cubman(calculation_type,first_cube,second_cube,output_cube,script_version="cubman-g16",all_formatted=True,scaling_factor=1.0):
    """
    Automatized interface function for using cubman
    Takes as input:
    - calculation_type can be any of the calculation types proposed by cubman
    - name of the first cube file (with extension)
    - name of the second cube file (with extension)
    - name of the output cube file (with extension)
    Outputs:
    - shell output lines
    - possible errors
    TODO:
    - [ ] add support for unformatted cube files
    - [ ] add support all calculation types 
        - [X] scale
        - [X] copy
        - [ ] ...
    """

    arguments=[script_version]
    process=subprocess.Popen(arguments,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)

    if all_formatted:
        if "sc" in calculation_type or "Sc" in calculation_type or "sC" in calculation_type:
            input_data="{}\n{}\ny\n{}\ny\n{}".format(calculation_type,first_cube,output_cube,scaling_factor)
        elif "co" in calculation_type or "Co" in calculation_type or "cO" in calculation_type:
            input_data="{}\n{}\ny\n{}\ny".format(calculation_type,first_cube,output_cube)
        else:
            input_data="{}\n{}\ny\n{}\ny\n{}\ny".format(calculation_type,first_cube,second_cube,output_cube)
    else:
        raise Exception("Sorry, unformatted input files are not supported yet")

    output,errors=process.communicate(input=input_data)
    return output,errors

def getTransitions(filename,root):
    filetag=filename.split(".")[0]
    tdlogfile=filetag+".log"
    with open(tdlogfile,"r") as f:
        lines=f.readlines()
    transitions=[]
    for i,line in enumerate(lines):
        if "Excited State" in line and " {}:".format(root) in line:
            c=1
            allTransitions=False
            while not allTransitions:
                if "->" in lines[i+c]:
                    jline=lines[i+c].replace("->","").split()
                    from_orbital=jline[0]
                    to_orbital=jline[1]
                    with_coeff=jline[2]
                    with_weight=str(2*float(with_coeff)**2)
                    transitions.append([from_orbital,to_orbital,with_coeff,with_weight])
                    c+=1
                else:
                    allTransitions=True
    transitions=np.array(transitions)
    weight_order=np.argsort(transitions[:,3].astype(float))[::-1]
    transitions=transitions[weight_order]
    return transitions

def canonical2transition_density_cube(filename,root,threshold=-1,cubman_version="cubman-g16",force_cubegen=True,force_recalc=True):
    filetag=filename.split(".")[0]
    fchkfile=filetag+".fchk"
    tdlogfile=filetag+".log"
    transitions=getTransitions(tdlogfile,root)

    logfile=open(filetag+"_transitionDensityCanonical.log","w")
    with open(filename,"r") as f:
        lines=f.readlines()
    for i,line in enumerate(lines):
        if "Number of electrons" in line:
            line=line.split()
            NElectrons=int(line[-1])
            NOccupied=NElectrons//2
        if "Orbital Energies" in line:
            line=line.split()
            NOrbitals=int(line[-1])
            orbital_energies=np.array([],dtype=object)
            for pline in lines[i+1:i+2+NOrbitals//5]:
                orbital_energies=np.append(orbital_energies,pline.split())
            orbital_energies=orbital_energies.flatten().astype(float)
    occupied_orbitals_energies=orbital_energies[:NOccupied]
    virtual_orbitals_energies=orbital_energies[NOccupied:2*NOccupied]
    
    from_orbital=transitions[:,0]
    to_orbital=transitions[:,1]
    with_coeff=transitions[:,2]
    with_weight=transitions[:,3]
    from_orbital=from_orbital[with_weight.astype(float)>threshold]
    to_orbital=to_orbital[with_weight.astype(float)>threshold]
    with_coeff=with_coeff[with_weight.astype(float)>threshold]
    with_weight=with_weight[with_weight.astype(float)>threshold]

    logfile.write("Threshold for selected pairs of orbitals: {}\n".format(threshold))
    logfile.write("Number of orbitals: {}\n".format(NOrbitals))
    logfile.write("Number of occupied orbitals: {}\n".format(NOccupied))
    logfile.write("Coefficients of the selected pairs of orbitals:\n {}\n".format(with_coeff))
    logfile.write("Weights of the selected pairs of orbitals:\n {}\n".format(with_weight))
    # Produce cube files
    for pair,coeff in enumerate(with_coeff):
        filetagPlus=filetag+"_occ_"+str(from_orbital[pair])
        filetagMinus=filetag+"_vir_"+str(to_orbital[pair])
        command="cubegen-g16 0 MO={} {} {}.cube 0 h".format(from_orbital[pair],filename,filetagPlus)
        logfile.write("Executed command: {}\n".format(command))
        if force_cubegen:
            os.system(command)
        command="cubegen-g16 0 MO={} {} {}.cube 0 h".format(to_orbital[pair],filename,filetagMinus)
        logfile.write("Executed command: {}\n".format(command))
        if force_cubegen:
            os.system(command)
        
        if force_recalc:
            filetagProd=filetag+"_pair_prod_"+str(pair)
            logfile.write("Produce cube {} from \n- cubefile {} PRODUCT WITH \n- cubefile {}\n".format(filetagProd+".cube",filetagPlus+".cube",filetagMinus+".cube"))
            output,error=auto_cubman("Sprod",filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube")
            print(error)

            filetagScaled=filetag+"_pair_scaled_"+str(pair)
            logfile.write("Scale cube {} with coeff {}\n".format(filetagProd+".cube",coeff))
            output,error=auto_cubman("Scale",filetagProd+".cube",filetagProd+".cube",filetagScaled+".cube",scaling_factor=coeff)
            print(error)

            filetagTransitionDensity=filetag+"_transitionDensity_upTo_"+str(pair)
            if pair==0:
                logfile.write("Produce transition density cube {} COPIED from first pair \n".format(filetagTransitionDensity+".cube"))
                output,error=auto_cubman("Copy",filetagScaled+".cube",filetagScaled+".cube",filetagTransitionDensity+".cube")
                print(error)
            else:
                filetagTransitionDensityPrevious=filetag+"_transitionDensity_upTo_"+str(pair-1)
                logfile.write("Produce transition density cube {} \n".format(filetagTransitionDensity+".cube"))
                output,error=auto_cubman("Add",filetagTransitionDensityPrevious+".cube",filetagScaled+".cube",filetagTransitionDensity+".cube")

            logfile.write("Deleting \n{}\n".format("\n".join([filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube",filetagScaled+".cube"])))
            os.system("rm {}".format(" ".join([filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube",filetagScaled+".cube"])))
            if pair!=0:
                logfile.write("Deleting \n{}\n".format(filetagTransitionDensityPrevious+".cube"))
                os.system("rm {}".format(filetagTransitionDensityPrevious+".cube"))
        elif not force_recalc:
            filetagProd=filetag+"_pair_prod_"+str(from_orbital[pair])
            filetagScaled=filetag+"_pair_scaled_"+str(to_orbital[pair])
            filetagTransitionDensity=filetag+"_transitionDensity_upTo_"+str(pair)
    logfile.close()
    return filetagTransitionDensity,occupied_orbitals_energies,virtual_orbitals_energies


def fchk2transition_density_cube(filename,cubman_version="cubman-g16",threshold=1/100,force_cubegen=True,force_recalc=True):
    filetag=filename.split(".")[0]
    logfile=open(filetag+"_transitionDensity.log","w")
    with open(filename,"r") as f:
        lines=f.readlines()
    for i,line in enumerate(lines):
        if "Number of electrons" in line:
            line=line.split()
            NElectrons=int(line[-1])
            NOccupied=NElectrons//2
        if "Orbital Energies" in line:
            line=line.split()
            NOrbitals=int(line[-1])
            orbital_energies=np.array([],dtype=object)
            for pline in lines[i+1:i+2+NOrbitals//5]:
                orbital_energies=np.append(orbital_energies,pline.split())
            orbital_energies=orbital_energies.flatten().astype(float)
    occupied_orbitals_energies=orbital_energies[:NOccupied]
    virtual_orbitals_energies=orbital_energies[NOccupied:2*NOccupied]
    
    weights=np.copy(occupied_orbitals_energies)[::-1]
    weights=weights[weights>threshold]
    logfile.write("Number of orbitals: {}\n".format(NOrbitals))
    logfile.write("Number of occupied orbitals: {}\n".format(NOccupied))
    logfile.write("Threshold for selected pairs of NTO: {}\n".format(threshold))
    logfile.write("Weights of the selected pairs of NTO:\n {}\n".format(weights))
    # Produce cube files
    for pair,weight in enumerate(weights):
        filetagPlus=filetag+"_NTO_occ_"+str(pair)
        filetagMinus=filetag+"_NTO_vir_"+str(pair)
        command="cubegen-g16 0 MO={} {} {}.cube 0 h".format(NOccupied-pair,filename,filetagPlus)
        logfile.write("Executed command: {}\n".format(command))
        if force_cubegen:
            os.system(command)
        command="cubegen-g16 0 MO={} {} {}.cube 0 h".format(NOccupied+1+pair,filename,filetagMinus)
        logfile.write("Executed command: {}\n".format(command))
        if force_cubegen:
            os.system(command)
        
        if force_recalc:
            filetagProd=filetag+"_NTO_prod_"+str(pair)
            logfile.write("Produce cube {} from \n- cubefile {} PRODUCT WITH \n- cubefile {}\n".format(filetagProd+".cube",filetagPlus+".cube",filetagMinus+".cube"))
            output,error=auto_cubman("Sprod",filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube")
            print(error)

            filetagScaled=filetag+"_NTO_scaled_"+str(pair)
            logfile.write("Scale cube {} with weight {}\n".format(filetagProd+".cube",weight))
            output,error=auto_cubman("Scale",filetagProd+".cube",filetagProd+".cube",filetagScaled+".cube",scaling_factor=weight)
            print(error)

            filetagTransitionDensity=filetag+"_transitionDensity_upTo_"+str(pair)
            if pair==0:
                logfile.write("Produce transition density cube {} COPIED from first pair \n".format(filetagTransitionDensity+".cube"))
                output,error=auto_cubman("Copy",filetagScaled+".cube",filetagScaled+".cube",filetagTransitionDensity+".cube")
                print(error)
            else:
                filetagTransitionDensityPrevious=filetag+"_transitionDensity_upTo_"+str(pair-1)
                logfile.write("Produce transition density cube {} \n".format(filetagTransitionDensity+".cube"))
                output,error=auto_cubman("Add",filetagTransitionDensityPrevious+".cube",filetagScaled+".cube",filetagTransitionDensity+".cube")

            logfile.write("Deleting \n{}\n".format("\n".join([filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube",filetagScaled+".cube"])))
            os.system("rm {}".format(" ".join([filetagPlus+".cube",filetagMinus+".cube",filetagProd+".cube",filetagScaled+".cube"])))
            if pair!=0:
                logfile.write("Deleting \n{}\n".format(filetagTransitionDensityPrevious+".cube"))
                os.system("rm {}".format(filetagTransitionDensityPrevious+".cube"))
        elif not force_recalc:
            filetagProd=filetag+"_NTO_prod_"+str(pair)
            filetagScaled=filetag+"_NTO_scaled_"+str(pair)
            filetagTransitionDensity=filetag+"_transitionDensity_upTo_"+str(pair)
    logfile.close()
    return filetagTransitionDensity,occupied_orbitals_energies,virtual_orbitals_energies

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        """
        Produce cube file for transition density and possibly plot it

        The program takes as inputs:
        - .fchk file of a NTO calculation 
        - a threshold for the number of NTO pairs to use for the re-construction of the transition density
        - rotation parameters for vmd surface plotting [require vmd]

        TODO list
        - [ ] 
        """
    )

    parser.add_argument("--filename",metavar="filename",required=True,help="Name (with fchk extension) of the file to be read",type=str)
    parser.add_argument("--root",metavar="root",required=True,help="Root for the targeted excited state",type=str)
    parser.add_argument("--workdir",metavar="workdir",required=False,help="(with fchk extension) of the file to be read",type=str)
    add_bool_arg(parser,'force_formchk',default=False,doc="Forcing (or not) the re-formatting of chk to fchk file")
    add_bool_arg(parser,'force_recalc',default=True,doc="Forcing (or not) the calc. of the transition density cubes from the fchk file")
    add_bool_arg(parser,'force_vmd',default=True,doc="Forcing (or not) the plot of MO cubes from via vmd [require vmd]")
    parser.add_argument("--threshold",metavar="threshold",required=False,help="threshold",default=0.010,type=float)
    parser.add_argument("--rx",metavar="rx",required=False,help="rx rotation",default=0.0,type=float)
    parser.add_argument("--ry",metavar="ry",required=False,help="ry rotation",default=0.0,type=float)
    parser.add_argument("--rz",metavar="rz",required=False,help="rz rotation",default=0.0,type=float)

    args=parser.parse_args()
    filename=args.filename
    root=args.root
    workdir=args.workdir
    force_formchk=args.force_formchk
    force_recalc=args.force_recalc
    force_vmd_plot=args.force_vmd
    threshold=args.threshold
    rx=args.rx
    ry=args.ry
    rz=args.rz

    filetag=filename.split(".")[0]

    if workdir is None:
        WORKDIR='cubes_'+filetag
    else:
        WORKDIR=workdir
    if not os.path.exists(WORKDIR):
        os.system("mkdir "+WORKDIR)

    print("Filename: ",filename)
    print("Directory: ",workdir)

    filetagTransitionDensity,occupied_orbitals_weights,virtual_orbitals_weights=canonical2transition_density_cube(filename,root,threshold=threshold,force_cubegen=force_recalc,force_recalc=force_recalc)
    filenameTransitionDensityCube=filetagTransitionDensity+".cube"
    if force_recalc:
        os.system("mv {} {}".format(filenameTransitionDensityCube,WORKDIR))

    with open(WORKDIR+"/MOs_generation.txt",'a+') as f:
        f.write("filename {}\n".format(filename))
        f.write("filename cube transition density {}\n".format(filenameTransitionDensityCube))
        f.write("rx={}, ry={}, rz={}\n".format(rx,ry,rz))
    if force_vmd_plot:
        os.system("VMDPATH=/usr/local/bin/vmd python3 ./vmd_cube.py ./"+WORKDIR+" --isovalue -0.0004 0.0004 --bright_scheme --opacity=0.7 --rx="+str(rx)+" --ry="+str(ry)+" --rz="+str(rz)+" --scale=1.5 --imagew=1000 --imageh=1000")

