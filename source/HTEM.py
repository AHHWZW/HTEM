#!/usr/bin/env python3

import os
import sys

from source.parameter import Basic_para, Input_para
from source.auto_gen import jobs, Input
from source.method import Common, Strain
from source.semi_analytic_model import SAM

# ----------------- some global constants -------------------------



# ----------------- some global parameters -------------------------
Na = Basic_para().Na
e = Basic_para().e
Ang = Basic_para().Ang
Bohr = Basic_para().Bohr
Ryd2eV = Basic_para().Ryd2eV
Ryd2GP = Basic_para().Ryd2GPa
ev2GP = Basic_para().ev2GPa
run_style_list = Basic_para().run_style_list
DMAX = Basic_para().DMAX
args = Input_para()

# MAIN

if __name__ == '__main__':
    command_line = True
    # Read the configuration file
    args.read_config()
    # Read the command line
    if (command_line):
        args.command_input()
    # Format the parameters
    args.format_para()
    # Read the configuration file
    config = jobs()
    # config = os.path.dirname(os.path.abspath(__file__))  
    config.read_potcar_config(config_dir)
    PATH = os.getcwd()
 
    # Check the runstyle
    if args.runstyle not in run_style_list:
        sys.stderr.write('Error: You must choose one of the following option to run this code!\n')
        sys.stderr.write(str(run_style_list)+'\n')
        os.sys.exit(1)

    # Read information from POSCAR file
    if os.path.exists('POSCAR'):
        input = Input()
        M, lattice, natom, sgn, volume, rho, cell = input.read_POSCAR()
        if args.lattice is None or args.sgn is None:
            args.lattice = lattice
            args.sgn = sgn
            args.write_config('lattice','sgn')
            sys.stdout.write('The symmetry of lattice and sgn are read from POSCAR file.\n')
        elif lattice is None and sgn is not None:
            args.lattice = Input().sgn_to_lat(args.sgn, args.cij_order)
            args.write_config('lattice', 'sgn')
            sys.stdout.write('The symmetry of lattice read form space group number.\n')
        elif lattice is not None and sgn is None:
            args.write_config('lattice')
        if args.natom is None:
            args.natom = natom
            args.write_config('natom')
            sys.stdout.write('The number of atoms for POSCAR is read from POSCAR file.\n')
        if args.M is None:
            args.M = M
            args.write_config('M')
            sys.stdout.write('The averaged relative atomic mass M is read from POSCAR file.\n')
    elif args.lattice is not None and args.M is not None and args.runstyle in ['model', 'anisotropy']:
        args.write_config('lattice','M')
        sys.stdout.write('The symmetry of lattice and M are read from the command line.\n')
    else:
        sys.stderr.write('Error: The POSCAR file is not exist. Please provide it first!\n')
        os.sys.exit(1)
    
    # Generate the input files for VASP
    if args.runstyle == 'gen':
        PBE_path = config.path_PBE
        if not os.path.exists('POTCAR'):
            input.POTCAR(cell, PBE_path)
            sys.stdout.write('The POTCAR file is generated according to the POSCAR file.\n')
        else:
            sys.stdout.write('The POTCAR file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        if not os.path.exists('KPOINTS'):
            if isinstance(args.ksetting, float):
                input.KSPACING_to_KPOINTS(cell, args.ksetting)
                sys.stdout.write('The KPOINTS file is generated.\n')
            elif isinstance(args.ksetting,list) and len(args.ksetting) == 3:
                input.KMESH_to_KPOINTS(args.ksetting)
                sys.stdout.write('The KPOINTS file is generated. The KPOINTS file is generated according to the KPOINTS setting.\n')
            else:
                sys.stderr.write('Error: The KPOINTS setting is not correct. Please check it!\n')
                os.sys.exit(1)
        if not os.path.exists('INCAR.relax1'):
            input.INCAR('relax1','metal',cell,config.cores,args)
            sys.stdout.write('The INCAR.relax1 file is generated.\n')
        else:
            sys.stdout.write('The INCAR.relax1 file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        if not os.path.exists('INCAR.relax2') and args.mode in ['cold','QSA']:
            input.INCAR('relax2','metal',cell,config.cores,args)
            sys.stdout.write('The INCAR.relax2 file is generated.\n')
        else:
            sys.stdout.write('The INCAR.relax2 file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        if not os.path.exists('INCAR.static') and args.mode in ['cold','QSA']:
            input.INCAR('static','metal',cell,config.cores,args)
            sys.stdout.write('The INCAR.static file is generated.\n')
        elif args.mode in ['cold','QSA']:
            sys.stdout.write('The INCAR.static file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        if not os.path.exists('INCAR.NPT') and args.mode in ['NPT']:
            input.INCAR('NPT','metal',cell,config.cores,args)
            sys.stdout.write('The INCAR.NPT file is generated.\n')
        elif args.mode in ['NPT']:
            sys.stdout.write('The INCAR.NPT file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        if not os.path.exists('INCAR.NVT') and args.mode in ['NVT']:
            input.INCAR('NVT','metal',cell,config.cores,args)
            sys.stdout.write('The INCAR.NVT file is generated.\n')
        elif args.mode in ['NVT']:
            sys.stdout.write('The INCAR.NVT file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
        args.write_config('mode','encut','ksetting','sigma')
        if args.mode in ['NPT','NVT']:
            args.write_config('MLFF')
        args.write_config('pressure','temperature')
        if not os.path.exists('state.in') and (args.inputfile == None or args.inputfile == 'state.in'):
            args.inputfile = 'state.in'
            input.state_in(args)
            sys.stdout.write('The state.in file is generated.\n')
            args.write_config('inputfile')
        elif os.path.exists('state.in'):
            sys.stdout.write('The state.in file is already exist and not changed. If you want to regenerate it, please delete it first.\n')
            args.write_config('inputfile')
        else:
            args.write_config('inputfile')
        
    os.chdir(PATH)
        
    #Steps for the elastic calculations
    if args.runstyle in ['create','strain', 'relax2', 'static','NVT','NPT']:
        # the first relax : ISIF = 4. Volume conservation, only relax shape.
        if args.runstyle in ['create']:
            Common().initial_structure_relax(args)
            if args.nvolume != None:
                args.write_config('inputfile','nvolume')
            else:
                args.write_config('inputfile','nstate')

        # creat the deformation structure required for energy-strain
        elif args.runstyle in ['strain']:
            if args.mode in ['NPT','NVT']:
                args.method = 'Stress'
            if args.mode != 'NPT':
                args.write_config('lattice','nstrain','maxstrain','method')
            else:
                args.write_config('lattice','method')
            Strain().strained_structure_create(args)
        # the second relax : ISIF = 2. Keep shape and volume, only relax iron.
        elif args.runstyle in ['relax2','static','NVT','NPT']:  #!!relax2和static缺少文件检查
            Common().elastic_calculation_preparation(DMAX, args)

        """
    # Cleanning of per-existing deformation structures.
    elif args.runstyle == 'clean':
        for i in range(args.nvolume):
            dirname = 'state_' + str(i+1)
            os.chdir(dirname)
            Common().clean_files()
            os.chdir('../')
        """

    # Obtain isothermal elastic stiff constants, moduli and sound speed.
    elif args.runstyle == 'get_results':
        Common().summary_results(args)
        args.write_config('forder')

    elif args.runstyle == 'anisotropy':
        if args.plt != None:
            args.write_config('plt')
        if args.read_file != None:
            args.write_config('read_file')
            if os.path.exists(args.read_file):
                Common().anisotropy(args, args.read_file)
            else:
                sys.stderr.write('Error: The file '+args.read_file+' is not exist!\n')
                os.sys.exit(1)
        else:
            if os.path.exists('Elasticity_T.dat'):
                Common().anisotropy(args, 'Elasticity_T.dat')
            if os.path.exists('Elasticity_S.dat'):
                Common().anisotropy(args, 'Elasticity_S.dat')
            if not os.path.exists('Elasticity_T.dat') and not os.path.exists('Elasticity_S.dat'):
                sys.stderr.write('Error: There is no Elasticity_T.dat file or Elasticity_S.dat file!\n')
                os.sys.exit(1)
    
    elif args.runstyle == 'model':
        if args.plt != None:
            args.write_config('plt')
        if args.read_file != None:
            args.write_config('read_file','weight','T_ref')
            if args.P_range != None:
                args.write_config('P_range')
            if args.T_range != None:
                args.write_config('T_range')
            if os.path.exists(args.read_file):
                Common().SAM_PT(args, args.read_file)
            else:
                sys.stderr.write('Error: The file '+args.read_file+' is not exist!\n')
                os.sys.exit(1)
        else:
            args.write_config('weight','T_ref')
            if args.P_range != None:
                args.write_config('P_range')
            if args.T_range != None:
                args.write_config('T_range')
            if os.path.exists('Elasticity_T.dat'):
                Common().SAM_PT(args, 'Elasticity_T.dat')
            if os.path.exists('Elasticity_S.dat'):
                Common().SAM_PT(args, 'Elasticity_S.dat')
            if not os.path.exists('Elasticity_T.dat') and not os.path.exists('Elasticity_S.dat'):
                sys.stderr.write('Error: There is no Elasticity_T.dat file or Elasticity_S.dat file!\n')
                os.sys.exit(1)
    
    elif args.runstyle == 'update':
        if args.upmode == 'CONTCAR':
            sys.stdout.write('Using the state_*/CONTCAR to update the inputfiles\n')
            Input().update_state_in(args, 'CONTCAR')
            args.inputfile = 'state.in'
            sys.stdout.write('The new inputfile is state.in\n')
        elif args.upmode == 'NPT':
            sys.stdout.write('Using the state_*/NPT/OUTCAR to update the inputfiles\n')
            Input().update_state_in(args, 'NPT')
            args.inputfile = 'state.in'
            sys.stdout.write('The new inputfile is state.in\n')
        args.write_config('inputfile','upmode')
    else:
        pass
