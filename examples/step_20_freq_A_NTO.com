%oldchk=step_20_freq_A.chk
%chk=step_20_freq_A_NTO.chk
%mem=16GB
%nprocshared=16
# pop=(NTO,saveNTO) density=(check,transition=3) cam-b3lyp/6-31+g(d) sym=com geom=allcheck guess=(read,only)

NTO orbitals for vertical transition 3 at geometry of step_20_freq_A

0 1

