import os
import numpy as np
#from ase.build import bulk
from ase.io import read
from spglib import get_symmetry_dataset
from .parameter import Basic_para
from .elasticity import Thermal
from .method import Common
import sys

class Input:
    """
    Input class for managing simulation parameters and generating input files for computational materials science calculations.

    Attributes:
        ISTART (int): Starting index for the calculation.
        ICHGARG (int): Charge density argument.
        LREAL (str): Real-space projection flag.
        PREC (str): Precision setting for calculations.
        LWAVE (str): Flag for wavefunction output.
        LCHARG (str): Flag for charge density output.
        ADDGRID (str): Flag for adding a grid.
        ENCUT (int): Energy cutoff value.
        GGA (str or None): Generalized gradient approximation.
        ISMEAR (int or None): Smearing method for electronic structure calculations.
        SIGMA (float or None): Width of the smearing.
        NELM (int): Maximum number of electronic steps.
        NELMIN (int): Minimum number of electronic steps.
        EDIFF (float): Energy convergence criterion.
        EDIFFG (float): Force convergence criterion.
        NSW (int or None): Number of ionic steps.
        IBRION (int or None): Ionic update algorithm.
        ISIF (int or None): Stress calculation flag.
        NPAR (int or None): Number of parallel processes.
        KPAR (int or None): K-point parallelization.
        ALGO (str): Algorithm for electronic structure calculations.
        INCAR_list (list): List of calculation types for which INCAR parameters can be initialized.

    Methods:
        init_for_system(type, system, args): Initializes INCAR parameters based on the type of calculation and system.
        extract_ENCUT(POTCAR): Extracts the ENCUT value from the POTCAR file.
        extract_LANGEVIN_GAMMA(POSCAR): Extracts LANGEVIN_GAMMA from the POSCAR file.
        extract_KPAR(num): Extracts KPAR based on the number of cores.
        INCAR(type, system, cell, ncores, args): Generates the INCAR file based on the provided parameters.
        POTCAR(cell, PBE_path): Generates the POTCAR file by merging individual POTCAR files for each element.
        KSPACING_to_KPOINTS(cell, KSPACING): Generates KPOINTS file from KSPACING.
        KMESH_to_KPOINTS(KMESH): Generates KPOINTS file from KMESH.
        POSCAR_supercell(cell, supercell): Generates a supercell POSCAR file.
        state_in(args): Generates a state.in file based on temperature and pressure ranges.
        update_state_in(args, mode): Updates the state.in file based on existing thermal data.
        read_POSCAR(): Reads information from the POSCAR file.
        sgn_to_lat(SGN, order): Converts space group number to lattice type.
    """
    
    def __init__(self):

        self.ISTART = 0
        self.ICHGARG = 2
        self.LREAL = "Auto"
        self.PREC = "Normal"
        self.LWAVE = ".False."
        self.LCHARG = ".False."
        self.ADDGRID = ".True."
        self.ENCUT = 400
        self.GGA = None
        #self.GGA = "PS"

        self.ISMEAR = None
        self.SIGMA = None
        self.NELM = 60
        self.NELMIN = 4
        self.EDIFF = 1E-6
        self.EDIFFG = -1E-2

        self.NSW = None
        self.IBRION = None
        self.ISIF = None

        self.NPAR = None
        self.KPAR = None
        self.ALGO = "Fast"

        self.AMIN = None
        self.MAXMIX = None
        self.IALGO = None
        self.POTIM = None
        self.NBLOCK = None
        self.KBLOCK = None
        self.TEBEG = None
        self.SMASS = None
        self.MDALGO = None
        self.PSTRESS = None
        self.LANGEVIN_GAMMA = None
        self.LANGEVIN_GAMMA_L = None

        self.INCAR_list = ['relax1','relax2','static','NVT','NPT']

    # initialize INCAR parameters for different types of calculations
    def init_for_system(self, type, system,args):
        """
        Initializes system parameters based on the specified type and system.
    
        This method configures various parameters for different types of calculations:
        - 'relax2': Sets parameters for relaxation with ISIF = 2.
        - 'relax1': Sets parameters for relaxation with ISIF = 3 or ISIF = 4, depending on the mode.
        - 'static': Configures parameters for static calculations.
        - 'NVT' or 'NPT': Sets parameters for Ab Initio Molecular Dynamics (AIMD) calculations.
    
        Parameters:
        type (str): The type of calculation ('relax2', 'relax1', 'static', 'NVT', 'NPT').
        system (str): The system type ('metal', 'semiconductor', or other).
        args: An object containing additional parameters such as pressure, temperature, and mode.

        Atrributes set:
        ISMEAR (int): The smearing method for electronic structure calculations.
        SIGMA (float): The width of the smearing.
        NSW (int): The number of ionic steps.
        IBRION (int): The ionic update algorithm.
        ISIF (int): The stress calculation flag.
        PSTRESS (int or str): The pressure stress tensor.
        NELM (int): The maximum number of electronic steps.
        IALGO (int): The algorithm for electronic structure calculations.
        POTIM (int): The time step for AIMD calculations.
        NELM (int): The maximum number of electronic steps.
        NBLOCK (int): The number of blocks for AIMD calculations.
        KBLOCK (int): The number of k-blocks for AIMD calculations.
        TEBEG (int): The beginning temperature for AIMD calculations.
        SIGMA (float): The width of the smearing for AIMD calculations.
        SMASS (int): The mass of the atoms for AIMD calculations.
        MDALGO (int): The algorithm for molecular dynamics calculations.
        LANGEVIN_GAMMA_L (int): The Langevin gamma value for AIMD calculations.
        ML_LMLFF (str): The machine learning force field flag.
    
        Raises:
        SystemExit: If temperature is not set for 'NVT' or 'NPT' calculations.
        """
        if type == "relax2":
            # Set parameters for relaxation with ISIF = 2# 
            if system == "metal":
                self.ISMEAR = 1
                self.SIGMA = 0.2
            elif system == "semiconductor":
                self.ISMEAR = -5
                self.SIGMA = None
            else:
                self.ISMEAR = 0
                self.SIGMA = 0.2
            self.NSW = 60
            self.IBRION = 2
            self.ISIF = 2
        elif type == "relax1":
            # Set parameters for relaxation with ISIF = 3 or ISIF=4#
            if system == "metal":
                self.ISMEAR = 1
                self.SIGMA = 0.2
            elif system == "semiconductor":
                self.ISMEAR = -5
                self.SIGMA = None
            else:
                self.ISMEAR = 0
                self.SIGMA = 0.2
            self.NSW = 60
            self.IBRION = 2
            if args.mode in ['cold','NPT']:
                self.ISIF = 3
                if args.pressure != None:
                    if isinstance(args.pressure, list):
                        self.PSTRESS = "pressure"
                    else:
                        self.PSTRESS = args.pressure*10
                else:
                    self.PSTRESS = 0
            else:
                self.ISIF = 4
            if args.pressure != None:
                self.ISIF = 3
                if isinstance(args.pressure, list):
                    self.PSTRESS = "pressure"
                else:
                    self.PSTRESS = args.pressure*10
        elif type == "static":
            # Set parameters for static calculations#
            if system == "metal":
                self.ISMEAR = -5
                self.SIGMA = None
            elif system == "semiconductor":
                self.ISMEAR = -5
                self.SIGMA = None
            else:
                self.ISMEAR = 0
                self.SIGMA = 0.2
            self.EDIFFG = None
            self.NSW = 0
            self.IBRION = -1
            self.ISIF = 2
        elif type == "NVT" or type == "NPT":
            # Set parameters for AIMD calculations#
            self.IBRION = 0
            self.IALGO = 48
            self.NSW = 10000
            self.POTIM = 1
            self.NELM = 60
            if type == "NVT":
                self.ISIF = 2
            elif type == "NPT":
                self.ISIF = 3
                if isinstance(args.pressure, list) or args.nstate != None:
                    self.PSTRESS = "pressure"
                else:
                    self.PSTRESS = args.pressure*10
            self.NBLOCK = 1
            #self.KBLOCK = 50
            if args.temperature == None:
                sys.exit("Temperature is not set, please check!")
            elif isinstance(args.temperature, list) or args.nstate != None:    
                self.TEBEG = "temperature"
                self.SIGMA  = "sigma"
            else:
                self.TEBEG = args.temperature
                self.SIGMA  = args.temperature / 11604.52511
            self.SMASS = None
            self.MDALGO = 3
            self.LANGEVIN_GAMMA_L = 100
            self.ISMEAR = -1
            
            if args.MLFF == True:
                self.NSW = 30000
                self.ML_LMLFF     = ".True" 
                self.ML_ISTART    = 0
                self.ML_IWEIGHT   = 1
                self.ML_MB        = 2500   
                self.ML_WTOTEN    = 0.0001
                self.ML_WTIFOR    = 0.01
                self.ML_RCUT1     = 8
        else:
            pass
    
    # extract ENCUT from POTCAR
    def extract_ENCUT(self, POTCAR):
        """
        Extracts the ENCUT value from a given POTCAR file.
    
        This method reads the POTCAR file line by line, searching for the line that contains 
        the "ENMAX" keyword. It extracts the corresponding value, multiplies it by 1.3, 
        and stores the maximum value found in the ENCUT attribute of the instance.
    
        Parameters:
        POTCAR (str): The path to the POTCAR file from which to extract the ENCUT value.

        Atrributes set:
        ENCUT (int): The maximum ENCUT value found in the POTCAR file.
        """
        # Extract ENCUT value from POTCAR file#
        ENCUT = []
        with open(POTCAR, 'r') as inputfile:
            for line in inputfile:
                if "ENMAX" in line:
                    ENCUT.append(int(float(line.split("=")[1].strip().split(";")[0].strip()) * 1.3))
        ENCUT = max(ENCUT)
        self.ENCUT = ENCUT

    # extract LANGEVIN_GAMMA from POSCAR
    def extract_LANGEVIN_GAMMA(self, POSCAR):
        """
        Extracts the LANGEVIN_GAMMA values from a given POSCAR file.
    
        This method reads the POSCAR file line by line and retrieves the number of elements 
        specified on the 8th line (index 7). It then generates a string of '100' values 
        corresponding to the number of elements, which is stored in the instance variable 
        LANGEVIN_GAMMA.
    
        Parameters:
        POSCAR (str): The path to the POSCAR file to be read.

        Atrributes set:
        LANGEVIN_GAMMA (str): A string of '100' values corresponding to the number of elements
        found in the POSCAR file.
        """
        with open(POSCAR, 'r') as inputfile:
            for i, line in enumerate(inputfile):
                if i == 7:
                    elements = line.strip().split()
                    num_elements = len(elements)
                    self.LANGEVIN_GAMMA = ' '.join(['100'] * num_elements)


    # extract KPAR from ncores
    def extract_KPAR(self, num):
        """
        Extracts the largest divisor of a given number.
    
        This method takes an integer input `num`, computes its divisors,
        and stores the maximum divisor found (excluding the number itself)
        in the instance variable KPAR.
    
        Parameters:
        num (int): The number of cores for parallel computation.

        Atrributes set:
        KPAR (int): The largest divisor of the input number.
        """
        num = int(num)
        self.KPAR = []
        self.KPAR = max([i for i in range(1, int(num**0.5) + 1) if num % i == 0])

    # generate INCAR file
    def INCAR(self, type, system, cell, ncores, args):
        """
        Generates an INCAR file for VASP simulations based on the specified parameters.
        
        The function initializes the simulation parameters, extracts necessary values from 
        related files, and writes the INCAR file with the appropriate settings for the 
        specified simulation type. It includes global, electronic, ionic, and other 
        parameters relevant to the simulation.
        
        Parameters:
        type (str): The type of simulation (e.g., 'NVT', 'NPT').
        system (str): The system type (default is set to 'metal').
        cell (object): The cell object used to obtain the chemical formula.
        ncores (int): The number of cores for parallel computation.
        args (object): Additional arguments containing simulation settings.

        Returns:
        None: The INCAR file is written to disk with the appropriate settings.
        """
        system = "metal"
        if type in self.INCAR_list:
            self.init_for_system(type, system, args)
        self.extract_ENCUT('POTCAR')
        if self.ENCUT < args.encut:
            self.ENCUT = args.encut
        self.extract_KPAR(ncores)
        if type in ['NVT','NPT']:
            self.extract_LANGEVIN_GAMMA('POSCAR')
        INCAR_name = 'INCAR.' + type
        with open(INCAR_name, 'w') as outputfile:
            outputfile.write("# INCAR for %s\n" % type)
            
            outputfile.write("# Global parameters\n")
            outputfile.write("SYSTEM = %s\n" % cell.get_chemical_formula())
            outputfile.write("ISTART = %d\n" % self.ISTART)
            outputfile.write("ICHARG = %d\n" % self.ICHGARG)
            outputfile.write("LREAL = %s\n" % self.LREAL)
            outputfile.write("PREC = %s\n" % self.PREC)
            outputfile.write("LWAVE = %s\n" % self.LWAVE)
            outputfile.write("LCHARG = %s\n" % self.LCHARG)
            outputfile.write("ADDGRID = %s\n" % self.ADDGRID)
            outputfile.write("ENCUT = %d\n" % self.ENCUT)
            if self.GGA != None:
                outputfile.write("GGA = %s\n" % self.GGA)
            
            outputfile.write("# Electronic parameters\n")
            outputfile.write("ISMEAR = %d\n" % self.ISMEAR)
            if self.SIGMA != None:
                outputfile.write("SIGMA = %s\n" % self.SIGMA)
            outputfile.write("NELM = %d\n" % self.NELM)
            outputfile.write("NELMIN = %d\n" % self.NELMIN)

            outputfile.write("# Ionic parameters\n")
            outputfile.write("EDIFF = %s\n" % self.EDIFF)
            if self.EDIFFG != None:
                outputfile.write("EDIFFG = %s\n" % self.EDIFFG)
            outputfile.write("NSW = %d\n" % self.NSW)
            outputfile.write("IBRION = %d\n" % self.IBRION)
            if self.PSTRESS != None and self.ISIF == 3:
                outputfile.write("PSTRESS = %s\n" % self.PSTRESS)
            outputfile.write("ISIF = %d\n" % self.ISIF)

            outputfile.write("# other parameters\n")
            if self.NPAR != None:
                outputfile.write("NPAR = %d\n" % self.NPAR)
            outputfile.write("KPAR = %d\n" % self.KPAR)
            outputfile.write("ALGO = %s\n" % self.ALGO)
            if self.AMIN != None:
                outputfile.write("AMIN = %s\n" % self.AMIN)
            if self.MAXMIX != None:
                outputfile.write("MAXMIX = %d\n" % self.MAXMIX)
            
            if type in ['NVT','NPT']:
                outputfile.write("# AIMD simulation parameters\n")
                outputfile.write("IALGO = %d\n" % self.IALGO)
                outputfile.write("POTIM = %s\n" % self.POTIM)
                outputfile.write("NBLOCK = %d\n" % self.NBLOCK)
                if self.KBLOCK != None:
                    outputfile.write("KBLOCK = %d\n" % self.KBLOCK)
                outputfile.write("TEBEG = %s\n" % self.TEBEG)
                #outputfile.write("SMASS = %d\n" % self.SMASS)
                outputfile.write("MDALGO = %d\n" % self.MDALGO)
                if self.LANGEVIN_GAMMA != None:
                    outputfile.write("LANGEVIN_GAMMA = %s\n" % self.LANGEVIN_GAMMA)
                if self.LANGEVIN_GAMMA_L != None:
                    outputfile.write("LANGEVIN_GAMMA_L = %d\n" % self.LANGEVIN_GAMMA_L)

            if args.MLFF == True and type in ['NVT','NPT']:
                outputfile.write("# Machine learning force field parameters\n")
                outputfile.write("ML_LMLFF = %s\n" % self.ML_LMLFF)
                outputfile.write("ML_ISTART = %d\n" % self.ML_ISTART)
                outputfile.write("ML_IWEIGHT = %d\n" % self.ML_IWEIGHT)
                outputfile.write("ML_MB = %d\n" % self.ML_MB)
                outputfile.write("ML_WTOTEN = %d\n" % self.ML_WTOTEN)
                outputfile.write("ML_WTIFOR = %d\n" % self.ML_WTIFOR)
                outputfile.write("ML_RCUT1 = %d\n" % self.ML_RCUT1)

    # generate POTCAR file
    def POTCAR(self, cell, PBE_path):
        """
        Merges POTCAR files for unique elements in a given cell.
    
        This method retrieves the POTCAR files for each unique chemical element 
        present in the provided cell. It checks for the existence of each POTCAR 
        file in the specified PBE path and raises an error if any file is missing. 
        The contents of the existing POTCAR files are then concatenated and written 
        to a single output file named 'POTCAR'.
    
        Parameters:
        cell (object): An object representing the cell containing chemical elements.
        PBE_path (str): The directory path where the POTCAR files for each element are located.

        Returns:
        None: The merged POTCAR file is written to disk.

        Raises:
        SystemExit: If any POTCAR file does not exist in the specified path.
        """
        POTCAR_list = []
        elements = list(dict.fromkeys(cell.get_chemical_symbols())) # Get unique elements in the cell
        for element in elements:
            POTCAR_path = PBE_path +"/" + element + '/POTCAR' 
            if not os.path.exists(POTCAR_path):
                sys.stderr.write("POTCAR for %s does not exist, please check!" % element)
                exit()
            else:
                POTCAR_list.append(POTCAR_path)
        POTCAR_merge = ''
        for i in range(len(POTCAR_list)):
            POTCAR = POTCAR_list[i]
            with open(POTCAR, 'r') as inputfile:
                POTCAR_merge += inputfile.read()

        # Write merged POTCAR file to disk   
        with open('POTCAR', 'w') as outputfile:
            outputfile.write(POTCAR_merge)
    
    # generate KPOINTS file from KSPACING
    def KSPACING_to_KPOINTS(self, cell, KSPACING):
        """
        Converts a specified KSPACING value into a corresponding KPOINTS mesh for a given crystal cell.
        
        This method calculates the reciprocal lattice vectors and determines the number of k-points 
        along each direction based on the provided KSPACING. It ensures that the number of k-points 
        is at least 1 in each direction. The resulting k-point mesh is then written to a file named 
        'KPOINTS' in the appropriate format for use in electronic structure calculations.

        Parameters:
        cell (object): An object representing the crystal cell, which contains the lattice vectors.
        KSPACING (float): The desired spacing between k-points in reciprocal space.

        Returns:
        None: The KPOINTS file is written to disk with the appropriate mesh.
        """
        b1 = 2*np.pi/np.linalg.norm(cell.cell[0])
        b2 = 2*np.pi/np.linalg.norm(cell.cell[1])
        b3 = 2*np.pi/np.linalg.norm(cell.cell[2])
        k_b1 = int(b1/KSPACING)
        k_b2 = int(b2/KSPACING)
        k_b3 = int(b3/KSPACING)

        # Ensure that the number of k-points is at least 1
        for i in k_b1, k_b2, k_b3:
            if i < 1:
                i = 1

        mesh = [k_b1, k_b2, k_b3]
        grid = 'Gamma'
        with open('KPOINTS', 'w') as outputfile:
            line = "Automatic mesh\n"
            outputfile.write(line)
            line = "0\n"
            outputfile.write(line)
            line = grid+"\n"
            outputfile.write(line)
            line = "{:d} {:d} {:d}\n".format(mesh[0], mesh[1], mesh[2])
            outputfile.write(line)
            line = "0 0 0\n"
            outputfile.write(line)
        pass
    
    # generate KPOINTS file from KMESH
    def KMESH_to_KPOINTS(self, KMESH):
        """
        Converts a given k-mesh into a KPOINTS file format for use in computational physics simulations.

        This method takes a list of three integers representing the k-point mesh dimensions and writes
        the corresponding KPOINTS file to disk. The file contains the appropriate header, grid type,
        and dimensions of the k-point mesh.
    
        Parameters:
        KMESH (list): A list containing three integers that define the k-point mesh dimensions.

        Returns:
        None: The KPOINTS file is written to disk with the appropriate mesh.
        """
        mesh = KMESH
        grid = 'Gamma'
        with open('KPOINTS', 'w') as outputfile:
            line = "Automatic mesh\n"
            outputfile.write(line)
            line = "0\n"
            outputfile.write(line)
            line = grid+"\n"
            outputfile.write(line)
            line = "{:d} {:d} {:d}\n".format(mesh[0], mesh[1], mesh[2])
            outputfile.write(line)
            line = "0 0 0\n"
            outputfile.write(line)
        pass
    
    # generate POSCAR file
#    def POSCAR(self, symmetry, element, lattice_constant):
#        if symmetry == "fcc" or symmetry == "bcc":
#            structure = bulk(element, symmetry, a=lattice_constant[0], b=lattice_constant[1], c=lattice_constant[2], cubic=True)
#        elif symmetry == "hcp":
#            structure = bulk(element, symmetry, a=lattice_constant[0], b=lattice_constant[1], c=lattice_constant[2])#, cubic=True)
#        structure.write('POSCAR', direct=True)
#        pass

    # generate supercell POSCAR
    def POSCAR_supercell(self, cell, supercell):
        """
        Generates a supercell from the given cell and specified dimensions.

        The function scales the original cell by the specified supercell dimensions and writes the resulting 
        supercell to a file named 'POSCAR-XYZ', where XYZ are the dimensions of the supercell.
    
        Parameters:
        cell (object): The original cell structure to be expanded into a supercell.
        supercell (list): A list of three integers representing the dimensions of the supercell in each direction.
        
        Returns:
        None: The supercell POSCAR file is written to disk.
        """
        supercell = [int(i) for i in supercell] # Ensure that the supercell dimensions are integers
        cell = cell*(supercell[0], supercell[1], supercell[2]) # Generate the supercell
        scname = 'POSCAR-' + str(supercell[0]) + str(supercell[1]) + str(supercell[2]) 
        cell.write(scname, direct=True)

    # generate state.in file
    def state_in(self, args):
        """
        Generates a state input file ('state.in') based on the specified temperature and pressure ranges for a given lattice type.
        
        The function reads the lattice parameters from a POSCAR file, determines the appropriate headers based on the lattice type,
        and writes the corresponding state data to 'state.in'. The output includes temperature, volume, pressure, density, 
        heat capacity, and lattice parameters as needed for different lattice types.

        Parameters:
        args: The arguments object containing the temperature and pressure ranges.

        Returns:
        None: The state.in file is written to disk with the appropriate state data.
        """
        M, lattice, natom, sgn, volume, rho, cell = self.read_POSCAR()
        para = {'T':'T(K)','V':'V(ang^3/atom)','P':'P(GPa)','rho':'rho(g/cm^3)','Cp':'Cp(J/mol/K)','a':'alpha_1(/K)','b':'alpha_2(/K)','c':'alpha_3(/K)'}
        header_list = ['T','V','P','rho','Cp']

        # Check if temperature and pressure are lists
        if isinstance(args.temperature, list):
            T_list = np.linspace(args.temperature[0], args.temperature[1], int(args.temperature[2]))
            nT = len(T_list)
        else:
            nT = 1
            if args.temperature == None:
                T = 0
            else:
                T = args.temperature
        if isinstance(args.pressure, list):
            P_list = np.linspace(args.pressure[0], args.pressure[1], int(args.pressure[2]))
            nP = len(P_list)
        else:
            nP = 1
            if args.pressure == None:
                P = 0
            else:
                P = args.pressure
        
        # Handle header for different lattice type
        if lattice == "C":
            header_list = header_list + ['a']
        elif lattice == "H":
            header_list = header_list + ['a','c']
        elif lattice == "TI":
            header_list = header_list + ['a','c']
        elif lattice == "TII":
            header_list = header_list + ['a','c']
        elif lattice == "RI":
            header_list = header_list + ['a','c']
        elif lattice == "RII":
            header_list = header_list + ['a','c']
        elif lattice == "O":
            header_list = header_list + ['a','b','c']
        elif lattice == "M":
            header_list = header_list + ['a','b','c']
        elif lattice == "N":
            header_list = header_list + ['a','b','c']
        else:
            pass

        # Write the header to state.in
        with open ('state.in', 'w') as outputfile:
            for i in header_list:
                outputfile.write(para[i].ljust(15))
            outputfile.write('\n')

        # Generate state data for temperature and pressure ranges
        for i in range(nT):
            if nT > 1:
                T = T_list[i]
            for j in range(nP):
                if nP > 1:
                    P = P_list[j]
                Cp, a, b, c = 0, 0, 0, 0

                if lattice == "C":
                    line = [T, volume, P, rho, Cp, a]
                elif lattice == "H":
                    line = [T, volume, P, rho, Cp, a, c]
                elif lattice == "TI":
                    line = [T, volume, P, rho, Cp, a, c]
                elif lattice == "TII":
                    line = [T, volume, P, rho, Cp, a, c]
                elif lattice == "RI":
                    line = [T, volume, P, rho, Cp, a, c]
                elif lattice == "RII":
                    line = [T, volume, P, rho, Cp, a, c]
                elif lattice == "O":
                    line = [T, volume, P, rho, Cp, a, b, c]
                elif lattice == "M":
                    line = [T, volume, P, rho, Cp, a, b, c]
                elif lattice == "N":
                    line = [T, volume, P, rho, Cp, a, b, c]
                else:
                    pass
                with open ('state.in', 'a') as outputfile:
                    for k in range(len(line)):
                        if k >= 5:
                            outputfile.write('{:<14.6e} '.format(line[k]))
                        else:
                            outputfile.write('{:<14.6f} '.format(line[k]))
                    outputfile.write('\n')
    
    def update_state_in(self, args, mode):
        """
        Updates the state information in the 'state.in' file based on the provided arguments and mode.
        
        The function reads thermal properties from the specified input file, generates state data for
        temperature and pressure ranges, and writes the results to 'state.in'. The header of the file
        is adjusted based on the lattice type, which determines the parameters included in the output.

        Parameters:
        args: The arguments object containing the temperature and pressure ranges.
        mode (str): The mode of the simulation ('CONTCAR' or 'NPT').

        Returns:
        None: The state.in file is updated with the new state data based on the existing thermal properties.
        """
        lattice = args.lattice
        thermal_old = Thermal()
        thermal_old.read_thermal_input(args.inputfile, args.lattice)
        T_list = thermal_old.T
        P_list = thermal_old.P
        Cp_list = thermal_old.Cp
        a_list = thermal_old.a
        b_list = thermal_old.b
        c_list = thermal_old.c

        para = {'T':'T(K)','V':'V(ang^3/atom)','P':'P(GPa)','rho':'rho(g/cm^3)','Cp':'Cp(J/mol/K)','a':'alpha_1(/K)','b':'alpha_2(/K)','c':'alpha_3(/K)'}
        header_list = ['T','V','P','rho','Cp']

        # Handle header for different lattice type
        if lattice == "C":
            header_list = header_list + ['a']
        elif lattice == "H":
            header_list = header_list + ['a','c']
        elif lattice == "TI":
            header_list = header_list + ['a','c']
        elif lattice == "TII":
            header_list = header_list + ['a','c']
        elif lattice == "RI":
            header_list = header_list + ['a','c']
        elif lattice == "RII":
            header_list = header_list + ['a','c']
        elif lattice == "O":
            header_list = header_list + ['a','b','c']
        elif lattice == "M":
            header_list = header_list + ['a','b','c']
        elif lattice == "N":
            header_list = header_list + ['a','b','c']
        else:
            pass
            
        # Write the header to state.in
        with open ("state.in", 'w') as outputfile:
            for i in header_list:
                outputfile.write(para[i].ljust(15))
            outputfile.write('\n')
        
        num = len(T_list)
        
        # Generate state data for temperature and pressure ranges
        for i in range(num):
            T = T_list[i]
            P = P_list[i]
            Cp = Cp_list[i]
            a = a_list[i]
            b = b_list[i]
            c = c_list[i]
            if mode == "CONTCAR":
                os.chdir('state_'+str(i+1))
                if os.path.exists('CONTCAR'):
                    volume = Common().get_volume('CONTCAR', "cell") / args.natom
                else:
                    sys.stderr.write('Error: CONTCAR for state_"+ str(i+1) +"/ does not exist, please check!\n')
                    os.sys.exit(1)
                os.chdir('..')
            elif mode == "NPT":
                os.chdir('state_'+str(i+1))
                os.chdir('NPT')
                if os.path.exists('OUTCAR'):
                    volume = Common().get_volume('OUTCAR', "NPT") / args.natom
                else:
                    sys.stderr.write('Error: OUTCAR for state_"+ str(i+1) +"/NPT/ does not exist, please check!\n')
                    os.sys.exit(1)
                os.chdir('../..')
            rho = args.M/(volume * Basic_para().Na/1E24)

            if lattice == "C":
                line = [T, volume, P, rho, Cp, a]
            elif lattice == "H":
                line = [T, volume, P, rho, Cp, a, c]
            elif lattice == "TI":
                line = [T, volume, P, rho, Cp, a, c]
            elif lattice == "TII":
                line = [T, volume, P, rho, Cp, a, c]
            elif lattice == "RI":
                line = [T, volume, P, rho, Cp, a, c]
            elif lattice == "RII":
                line = [T, volume, P, rho, Cp, a, c]
            elif lattice == "O":
                line = [T, volume, P, rho, Cp, a, b, c]
            elif lattice == "M":
                line = [T, volume, P, rho, Cp, a, b, c]
            elif lattice == "N":
                line = [T, volume, P, rho, Cp, a, b, c]
            else:
                pass
            with open ('state.in', 'a') as outputfile:
                for k in range(len(line)):
                    if k >= 5:
                        outputfile.write('{:<14.6e} '.format(line[k]))
                    else:
                        outputfile.write('{:<14.6f} '.format(line[k]))
                outputfile.write('\n')

    # read POSCAR information
    def read_POSCAR(self):
        """
        Reads the POSCAR file and calculates various properties of the crystal structure.

        This method reads the POSCAR file and extracts the atomic masses, lattice parameters,
        number of atoms, symmetry number, volume per atom, and density of the crystal structure.

        Returns:
        M (float): The total mass of the atoms in the cell.
        lattice (str): The lattice type of the crystal structure.
        natom (int): The number of atoms in the cell.
        sgn (int): The space group number of the crystal structure.
        volume (float): The volume per atom of the crystal structure.
        rho (float): The density of the crystal structure.
        cell (object): The cell object containing the crystal structure.

        Raises:
        SystemExit: If the POSCAR file does not exist.
        """
        Na = Basic_para().Na
        cell= read('POSCAR')
        atoms = cell.get_chemical_symbols()
        volume = cell.get_volume()/len(atoms)
        lat = (cell.get_cell(),cell.get_scaled_positions(),cell.get_atomic_numbers())
        try:
            sgn = get_symmetry_dataset(lat).number
        except:
            sgn = get_symmetry_dataset(lat)['number']
        lattice = self.sgn_to_lat(int(sgn), 2) 
        M = sum(cell.get_masses())/len(atoms)
        rho = M/(volume * Na/1E24)
        return M, lattice, len(atoms), sgn, volume, rho, cell
    
    # convert space group number to lattice type
    def sgn_to_lat(self, SGN, order=2):
        """
        Converts a given space group number (SGN) to its corresponding 
        lattice type (LC) based on the specified order.
    
        The function categorizes the space group number into various 
        lattice types including Triclinic, Monoclinic, Orthorhombic, 
        Tetragonal, Rhombohedral, Hexagonal, and Cubic, and returns 
        the appropriate lattice type based on the defined ranges.
    
        Parameters:
        SGN (int): The space group number to be converted.
        order (int, optional): The order of the conversion, default is 2.
    
        Returns:
        str: The lattice type corresponding to the given space group number.
        """
        if (1 <= SGN and SGN <= 2):      # Triclinic
            LC = 'N'
            if (order == 2):
                ECs = 21
        elif(3 <= SGN and SGN <= 15):    # Monoclinic
            LC = 'M'
            if (order == 2):
                ECs = 13
        elif(16 <= SGN and SGN <= 74):   # Orthorhombic
            LC = 'O'
            if (order == 2):
                ECs = 9
        elif(75 <= SGN and SGN <= 88):   # Tetragonal II
            LC = 'TII'
            if (order == 2):
                ECs = 7
        elif(89 <= SGN and SGN <= 142):  # Tetragonal I
            LC = 'TI'
            if (order == 2):
                ECs = 6
        elif(143 <= SGN and SGN <= 148):  # Rhombohedral II
            LC = 'RII'
            if (order == 2):
                ECs = 7
        elif(149 <= SGN and SGN <= 167):  # Rhombohedral I
            LC = 'RI'
            if (order == 2):
                ECs = 6
        elif(168 <= SGN and SGN <= 194):  # Hexagonal
            LC = 'H'
            if (order == 2):
                ECs = 5
        elif(195 <= SGN and SGN <= 230):  # Cubic
            LC = 'C'
            if (order == 2):
                ECs = 3
        else:
            sys.exit(
                '\n     ... Oops ERROR: WRONG Space-Group Number !?!?!?    \n')
        return LC

class jobs:
    """
    A class representing a job configuration for a job scheduler.

    Attributes:
        code (str): The code associated with the job.
        cores (int): The number of cores to allocate for the job.
        nodes (int): The number of nodes to allocate for the job.
        partion (str): The partition to submit the job to.
        jobname (str): The name of the job.
        errorname (str): The name of the file to log errors.
        output (str): The name of the file to log output.
        time (str): The time limit for the job.
        load_env (str): The environment setup commands to run before the job.
        command (str): The command to execute for the job.
        path_PBE (str): Path to the PBE potential files.
        path_LDA (str): Path to the LDA potential files.

    Methods:
        read_potcar_config(path): Reads the VASP configuration from a specified path and sets the corresponding attributes.
        create(name): Generates a job submission script based on the job attributes (currently commented out).
        submit(): Placeholder method for submitting the job (currently unimplemented).
    """
    def __init__(self):

        self.code = None
        self.cores = None
        self.nodes = None
        self.partion = None
        self.jobname = None
        self.errorname = None
        self.output = None
        self.time = None
        self.load_env = None
        self.command = None
        self.path_PBE = None
        self.path_LDA = None
    
    # read vasp.config file
    def read_potcar_config(self,path):
        """
        Reads the VASP configuration from a specified path and sets the corresponding attributes.
    
        This method looks for a file named 'vasp.config' in the provided directory path. 
        If the file exists, it reads each line, splits it into a parameter and its value, 
        and assigns the value to the corresponding attribute of the class instance.
    
        Parameters:
        path (str): The directory path where the 'vasp.config' file is located.
    
        Returns:
        None: The method modifies the instance attributes directly.

        Raises:
        FileNotFoundError: If the 'vasp.config' file is not found in the specified path.
        """
        config = path+"/vasp.config"
        if os.path.exists(config):
            with open(config, 'r') as inputfile:
                for line in inputfile:
                    para = line.split("=")[0].strip()
                    value = line.split("=")[1].strip('\n')
                    setattr(self, para, value)
        else:
            pass
    """
    def create(self, name):
        job_name = "submit_"+name+".sh"
        with open(job_name, 'w') as outputfile:
            line = "!/bin/bash\n\n"
            outputfile.write(line)
            if self.cores != None:
                line = "#SBATCH -n "+self.cores+'\n'
                outputfile.write(line)
            else:
                pass
            if self.nodes != None:
                line = "#SBATCH --nodes="+self.nodes+'\n'
                outputfile.write(line)
            else:
                pass
            if self.partion != None:
                line = "#SBATCH --partition="+self.partion+'\n'
                outputfile.write(line)
            else:
                pass
            if self.jobname != None:
                line = "#SBATCH --job-name="+self.jobname+'\n'
                outputfile.write(line)
            else:
                pass
            if self.errorname != None:
                line = "#SBATCH --error="+self.errorname+'\n'
                outputfile.write(line)
            else:
                pass
            if self.output != None:
                line = "#SBATCH --output="+self.output+'\n'
                outputfile.write(line)
            else:
                pass
            if self.load_env != None:
                line = self.load_env+'\n'
                outputfile.write(line)
            else:
                pass
            if self.command != None:
                line = self.command+'\n'
                outputfile.write(line) 

    def submit(self):

        pass
    """