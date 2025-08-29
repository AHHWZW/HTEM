import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import shutil
import glob
import warnings
import re
from collections import deque

from scipy import interpolate
from .elasticity import Elasticity, Thermal
from .parameter import Basic_para
from .write_output import write_output
from .anisotropy import Anisotropy
from .semi_analytic_model import SAM

class Common:
    """
    Common class provides methods for various computational tasks related to material properties, including file checks, energy and stress extraction, volume and pressure calculations, and elastic property computations.

    Methods:
    - file_check(mode): Verifies the existence of required files based on the specified mode.
    - extract_energy(fin='OSZICAR'): Extracts energy values from the OSZICAR file.
    - extract_stress(fin='OUTCAR', mode='cold', MLFF=False): Extracts stress information from the OUTCAR file.
    - extract_volume_MD(fin='OUTCAR'): Extracts the volume of the cell from the OUTCAR file.
    - extract_pressure_MD(fin='OUTCAR'): Extracts total pressure from the OUTCAR file.
    - extract_temperature_MD(fin='OUTCAR'): Extracts temperature information from the OUTCAR file.
    - extract_pressure(fin='OUTCAR'): Extracts external pressure from the OUTCAR file.
    - calc_CV(strain, stress, fitres): Calculates the coefficient of variation for given strain and stress data.
    - get_all_mod(C, units='eV', use_symmetry=True, method='Energy'): Computes elastic properties based on the provided parameters.
    - strlist2flist(strlist): Converts a list of strings to a list of floats.
    - initial_structure_relax(args): Prepares for initial structure relaxation based on input arguments.
    - get_volume(input='POSCAR', mode='cell'): Retrieves the volume from the specified input file.
    - change_POSCAR(thermal_input, n, na, mode): Modifies the POSCAR file based on calculated volumes or states.
    - make_dirfile(n): Creates directories and copies necessary files for calculations.
    - update_INCAR(args): Updates INCAR files based on the provided arguments.
    - change_INCAR_para(parameter, value, fn): Changes specified parameters in the INCAR file.
    - elastic_calculation_preparation(DMAX, args): Prepares for elastic calculations by setting up directories and files.
    - trans_isentroestc(C_S, thermal_input, i): Converts isothermal elastic constants to isentropic elastic constants.
    - summary_results(args): Summarizes elastic results across different states.
    - get_cell_matrix(input='POSCAR'): Retrieves the cell matrix from the POSCAR file.
    - get_one_state_results(args, C, i): Obtains results for one state based on the specified arguments.
    - read_stress(fin='stress.out', mode='cold'): Reads stress information from the specified file.
    - anisotropy(args, fn): Analyzes anisotropy information based on the specified modulus.
    - SAM_PT(args, fn): Models elastic properties at specified temperatures and pressures.
    """
    
    # file check
    def file_check(self, mode):
        """
        Checks for the existence of required files in the current directory based on the specified mode.
    
        This method verifies the presence of the following files:
        - POTCAR
        - POSCAR
        - INCAR.relax1
        - INCAR.relax2 (if mode is 'cold' or 'QSA')
        - INCAR.static (if mode is 'cold' or 'QSA')
        - INCAR.NPT (if mode is 'NPT')
        - INCAR.NVT (if mode is 'NVT')
    
        If any of the required files are missing, an error message is written to standard error,
        and the program exits with a status code of 1.
    
        Parameters:
        mode (str): The mode of operation which determines the required files.
        """
        #check if POTCAR and POSCAR are existing in
        #current directory.
        if not os.path.exists("POTCAR"):
            sys.stderr.write("Error: POTCAR is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("POSCAR"):
            #To be done, add INCAR.relax2 for NVT:
            sys.stderr.write("Error: POSCAR is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("INCAR.relax1"):
            sys.stderr.write("Error: INCAR.relax1 is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("INCAR.relax2") and mode in ['cold', 'QSA']:
            sys.stderr.write("Error: INCAR.relax2 is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("INCAR.static") and mode in ['cold', 'QSA']:
            sys.path.exists("Error: INCAR.static is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("INCAR.NPT") and mode in ['NPT']:
            sys.stderr.write("Error: INCAR.NPT is not found!\n")
            os.sys.exit(1)
        if not os.path.exists("INCAR.NVT") and mode in ['NVT']:
            sys.stderr.write("Error: INCAR.NVT is not found!\n")
            os.sys.exit(1)

    # get energy information
    def extract_energy(self, fin='OSZICAR'):
        """
        Extracts the energy value from the specified OSZICAR file.
    
        This method checks for the existence of the OSZICAR file in the current directory.
        If the file exists, it reads the last line and extracts the energy value (F=).
        If the file does not exist, it outputs an error message and exits the program.
    
        Parameters:
        fin (str): The name of the OSZICAR file to read. Defaults to 'OSZICAR'.
    
        Returns:
        float: The extracted energy value.
    
        Raises:
        SystemExit: If the OSZICAR file does not exist.
        """
        # get energy results in each directory.
        if os.path.exists(fin):
            with open(fin,'r') as file:
                lines = file.readlines()
            last_line = lines[-1]
            eng = float(last_line.split()[2])  # use F=
        else:
            sys.stderr.write('Error: OSZICAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!')
            os.sys.exit(1)
        return eng

    # get stress information
    def extract_stress(self, fin='OUTCAR', mode='cold', MLFF=False):
        """
        Extracts stress information from the specified input file (OUTCAR) and writes it to an output file (stress.out).
        
        Parameters:
        fin (str): The name of the input file. Default is 'OUTCAR'.
        mode (str): The mode of extraction. Options are 'cold', 'QSA', 'NVT', or 'NPT'. Default is 'cold'.
        MLFF (bool): A flag indicating whether to use machine learning force fields. Default is False.
        
        Raises:
        FileNotFoundError: If the specified input file does not exist.
        
        The function processes the input file based on the specified mode:
        - In 'cold' or 'QSA' mode, it extracts lines containing stress information.
        - In 'NVT' or 'NPT' mode, it extracts total kinetic energy information, with special handling if MLFF is True.
        """
        if fin == 'OUTCAR':
            if os.path.exists(fin):
                with open(fin, 'r') as inputfile, open('stress.out', 'w') as outputfile:
                    if mode == 'cold' or mode == 'QSA':
                        for line in inputfile:
                            if 'in kB' in line:
                                line = line.replace('in kB', '')
                                outputfile.write(line)
                    elif mode == 'NVT' or mode == 'NPT':
                        if MLFF == True:
                            queue = deque(maxlen=2)
                            for line in inputfile:
                                queue.append(line)
                                if 'Total+kin.' in queue[0] and 'volume of cell' in queue[1]:
                                    queue[0] = queue[0].replace('Total+kin.', '')
                                    outputfile.write(queue[0])
                        else:
                            for line in inputfile:
                                if 'Total+kin.' in line:
                                    line = line.replace('Total+kin.', '')
                                    outputfile.write(line)
            elif os.path.exists('stress.out'):
                pass
            else:
                sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')

    # get volume information for NPT calculation
    def extract_volume_MD(self, fin='OUTCAR'):
        """
        Extracts the volume of the cell from the specified OUTCAR file.
    
        This method reads the OUTCAR file line by line, searching for the line that contains
        the phrase 'volume of cell'. When found, it extracts the volume value and writes it
        to a file named 'volume.out'. If the OUTCAR file does not exist, an error message
        is printed to standard error.
    
        Parameters:
        fin (str): The path to the OUTCAR file. Defaults to 'OUTCAR'.
    
        Raises:
        FileNotFoundError: If the OUTCAR file does not exist.
        """
        if os.path.exists(fin):
            with open(fin, 'r') as inputfile, open('volume.out', 'w') as outputfile:
                queue = deque(maxlen=2)
                for line in inputfile:
                    queue.append(line)
                    if 'Total+kin.' in queue[0] and 'volume of cell' in queue[1]:
                        queue[1] = queue[1].replace('volume of cell :', '')
                        outputfile.write(queue[1])
        elif os.path.exists('volume.out'):
            pass
        else:
            sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')
    
    # get pressure information for NPT calculation
    def extract_pressure_MD(self, fin='OUTCAR'):
        """
        Extracts the total pressure from the specified OUTCAR file and writes it to an output file.
    
        Parameters:
        fin (str): The path to the OUTCAR file. Defaults to 'OUTCAR'.
    
        Raises:
        FileNotFoundError: If the OUTCAR file does not exist, an error message is printed to stderr.
    
        The function reads through the OUTCAR file line by line, searching for the line that contains
        'total pressure'. It extracts the pressure value, converts it from bar to MPa (by dividing by 10),
        and writes the result to 'pressure.out'.
        """
        if os.path.exists(fin):
            with open(fin, 'r') as inputfile, open('pressure.out', 'w') as outputfile:
                for line in inputfile:
                    if 'total pressure' in line:
                        pressure_temp = float(line.split()[3])/10
                        outputfile.write(str(pressure_temp)+'\n')
        elif os.path.exists('pressure.out'):
            pass
        else:
            sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')

    # get temperature information for NPT calculation
    def extract_temperature_MD(self, fin='OUTCAR'):
        """
        Extracts the kinetic temperature from the specified OUTCAR file.
    
        This method reads the OUTCAR file to find the line containing the kinetic lattice energy 
        (EKIN_LAT) and extracts the corresponding temperature value. The extracted temperature 
        is then written to a file named 'temperature.out'. If the OUTCAR file does not exist, 
        an error message is printed to standard error.
    
        Parameters:
        fin (str): The path to the OUTCAR file. Defaults to 'OUTCAR'.
    
        Raises:
        FileNotFoundError: If the specified OUTCAR file does not exist.
        """
        if os.path.exists(fin):
            with open(fin, 'r') as inputfile, open('temperature.out', 'w') as outputfile:
                for line in inputfile:
                    if 'kin. lattice  EKIN_LAT=' in line:
                        temperature_temp = float(line.split()[5])
                        outputfile.write(str(temperature_temp)+'\n')
        elif os.path.exists('temperature.out'):
            pass
        else:
            sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')
    
    # get external pressure information
    def extract_pressure(self, fin='OUTCAR'):
        """
        Extracts the external pressure from the specified OUTCAR file.
    
        This method checks for the existence of the OUTCAR file, reads its contents,
        and retrieves the last occurrence of the external pressure value. The pressure
        is returned in units of GPa (converted from the original units).
    
        Parameters:
        fin (str): The path to the OUTCAR file. Defaults to 'OUTCAR'.
    
        Returns:
        float: The external pressure in GPa.
    
        Raises:
        SystemExit: If the OUTCAR file does not exist, an error message is printed
                     and the program exits.
        """
        # get pressure results in each directory.
        if os.path.exists(fin):
            with open(fin,'r') as file:
                lines = file.readlines()
            external_line = [line for line in lines if 'external' in line]
            last_line = external_line[-1]
            P = float(last_line.split()[3])/10  #!! use external pressure 
        elif os.path.exists('pressure.out'):
            pass
        else:
            sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')
            os.sys.exit(1)
        return P
    
    # fit
    def calc_CV(self, strain, stress, fitres):
        """
        Calculate the coefficient of variation (CV) for given strain and stress data.
    
        This method computes the CV based on the difference between the actual stress values
        and the fitted stress values obtained from a polynomial fit. It is applicable only when
        the number of data points (ndata) is greater than 2.
    
        Parameters:
        strain (array-like): An array of strain values.
        stress (array-like): An array of corresponding stress values.
        fitres (array-like): Coefficients of the polynomial used for fitting the stress data.
    
        Returns:
        float: The calculated coefficient of variation (CV).
        """
        # useful only for ndata > 2
        S = 0
        for k in range(len(strain)):
            Y = stress[k]
            Yfit = np.polyval(fitres, strain[k])
            S = S + (Yfit - Y)**2
        CV = np.sqrt(S / len(strain))
        return CV

    # get elastic properties
    def get_all_mod(self, C, units='eV', use_symmetry = True, method = 'Energy'):
        """
        Computes the complete elastic modulus matrix for a given material.
    
        Parameters:
        C (object): An object containing the material properties, including C_matrix and pressure P.
        units (str): The units for the elastic modulus ('eV', 'Mbar', or 'kbar'). Default is 'eV'.
        use_symmetry (bool): A flag indicating whether to enforce symmetry in the modulus matrix. Default is True.
        method (str): The method to adjust the modulus matrix, currently supports 'Energy'. Default is 'Energy'.
    
        The function modifies the C_matrix based on the specified units and method, and optionally applies symmetry.
        """
        if units == 'eV':
            C.C_matrix *= Basic_para().ev2GPa / C.V
        elif units == 'Mbar':
            C.C_matrix *= 100
        elif units == 'kbar':
            C.C_matrix *= 0.1
        if method == 'Energy':
            C.C_matrix[0, 1] += C.P
            C.C_matrix[0, 2] += C.P
            C.C_matrix[1, 2] += C.P
            C.C_matrix[3, 3] -= C.P/2
            C.C_matrix[4, 4] -= C.P/2
            C.C_matrix[5, 5] -= C.P/2
        C.C_matrix[1, 0]  = C.C_matrix[0, 1] 
        C.C_matrix[2, 0]  = C.C_matrix[0, 2] 
        C.C_matrix[2, 1]  = C.C_matrix[1, 2] 
 
        if use_symmetry:
            for i in range(5):
                for j in range(i + 1, 6):
                    C.C_matrix[j, i] = C.C_matrix[i, j]
    
    # convert string list to float list
    def strlist2flist(self, strlist):
        """
        Converts a list of strings to a list of floats.
    
        Parameters:
        strlist (list of str): A list containing string representations of numbers.
    
        Returns:
        list of float: A list containing the corresponding float values of the input strings.
        
        Raises:
        ValueError: If any string in strlist cannot be converted to a float.
        """
        flist = []
        for s in strlist:
            flist.append(float(s))
        return flist
    
    # initial structure relaxation
    def initial_structure_relax(self,args):
        """
        Initializes the structure relaxation process based on the provided arguments.
    
        This method performs the following steps:
        1. Checks the validity of the input mode.
        2. Reads thermal input data from the specified input file and lattice.
        3. Depending on whether a specific volume is provided, it either:
           - Creates a directory for the specified number of volumes and updates the POSCAR file accordingly.
           - Determines the number of states from the thermal input and creates a directory for that number, updating the POSCAR file for states.
        4. Updates the INCAR file with the provided arguments.
    
        Parameters:
        args: An object containing the necessary parameters for the relaxation process, including mode, input file, lattice, number of volumes, number of states, and number of atoms.
        """
        self.file_check(args.mode)
        thermal_input = Thermal()
        thermal_input.read_thermal_input(args.inputfile, args.lattice)
        if args.nvolume != None:
            self.make_dirfile(args.nvolume)
            self.change_POSCAR(thermal_input,args.nvolume,args.natom, "volume")
        else:
            args.nstate = len(thermal_input.V)
            self.make_dirfile(args.nstate)
            self.change_POSCAR(thermal_input,args.nstate,args.natom, "state")
        self.update_INCAR(args) 

    # get volume
    def get_volume(self, input='POSCAR', mode = "cell"):
        """
        Calculate the volume of a cell based on the specified mode.
    
        Parameters:
        input (str): The input file name, default is 'POSCAR'.
        mode (str): The mode of calculation, either 'cell' or 'NPT'. Default is 'cell'.
    
        Returns:
        float: The calculated volume of the cell.
    
        If mode is 'cell', the function reads the scale from the second line of the input file,
        computes the cell matrix, and calculates the volume using the determinant of the matrix
        scaled by the cube of the scale factor.
    
        If mode is 'NPT', the function executes a shell command to extract the volume from the
        input file and computes the average volume from the extracted values.
        """
        if mode == "cell":
            lines = open(input, 'r').readlines()
            cnt = 1
            for line in lines:
                if cnt == 2:
                    res = line.split()
                    scale = float(res[0])
                    break
                cnt += 1
            if scale < 0:
                vol = -scale
            else:
                cellm = self.get_cell_matrix(input)
                vol = np.linalg.det(cellm) * scale**3
        elif mode == "NPT":
            if os.path.exists(input):
                os.system("grep 'volume of cell' "+input+" | awk '{print $5}' > volume.out")
                vol_list = np.loadtxt('volume.out')
                vol = np.average(vol_list)
                os.remove('volume.out')
            elif os.path.exists('volume.out'):
                vol_list = np.loadtxt('volume.out')
                vol = np.average(vol_list)
            else:
                sys.stderr.write('Error: OUTCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')
                os.sys.exit(1)
        return vol

    # change POSCAR
    def change_POSCAR(self, thermal_input, n, na, mode):
        """
        Changes the POSCAR file based on the specified mode ('volume' or 'state') 
        and calculates the corresponding values for a given number of states.
    
        Parameters:
        thermal_input: An object containing properties V (volume), T (temperature), 
                       and P (pressure).
        n (int): The number of states to calculate.
        na (float): A scaling factor for the calculated volumes or states.
        mode (str): The mode of operation, either 'volume' or 'state'.
    
        Raises:
        SystemExit: If the number of states in the input is less than n in 'volume' mode.
    
        Returns:
        None: This method updates the POSCAR file in each state directory.

        Notes:
        The mode volume is not used in this program.
        """
        V = thermal_input.V
        T = thermal_input.T
        P = thermal_input.P
        if mode == "volume":
            if len(V) == n:
                calv_list_input = V * na
            elif len(V) < n:
                calv_list_input = V * na
                sys.stderr.write('Error: The number of states in state.in is less than nvolume, please check!')
                os.sys.exit(1)
            else:
                vmax = max(V)
                vmin = min(V)
                calv_list_input = np.linspace(vmin, vmax, n) * na
        elif mode == "state":
            calv_list_input = V * na
        
        POSCAR_volume = self.get_volume('POSCAR', "cell")
        calv_list_input = (calv_list_input / POSCAR_volume) ** (1/3)

        for i in range(n):
            dirname = 'state_' + str(i+1)
            os.chdir(dirname)
            fw = open(r'POSCAR','r+',encoding = "utf-8")
            flist = fw.readlines()
            #flist[1] = "-" + str(calv_list_input[i]) + "\n"
            flist[1] = str(calv_list_input[i]) + "\n"
            fw = open(r'POSCAR','w+',encoding = "utf-8")
            fw.writelines(flist)
            fw.close()
            os.chdir('../')
        if mode == "volume":
            sys.stdout.write(f"The calculated volumes are \n {calv_list_input/na} ang^3/atom\n")
        elif mode == "state":
            sys.stdout.write(f"The calculated states are at \n {T} K\n")
            sys.stdout.write(f"The calculated states are at \n {P} GPa\n")

    # create directory and copy files
    def make_dirfile(self, n):
        """
        Creates a specified number of state directories and populates them with necessary files.
    
        This method first checks for existing state directories (named 'state_*') and removes them if found,
        issuing a warning to the user. It then creates 'n' new directories named 'state_1', 'state_2', ..., 
        'state_n'. For each directory, it copies the files 'POSCAR', 'POTCAR', and optionally 'KPOINTS' 
        from the parent directory, as well as 'INCAR.relax1'. The method changes the current working 
        directory to each new state directory during the copying process.
    
        Parameters:
        n (int): The number of state directories to create.

        Returns:
        None: This method creates directories and copies files.

        Notes:
        The method issues a warning if existing state directories are found and removes them.
        """
        folders = glob.glob('state_*') 
        if len(folders) != 0:
            for folder in folders:
                try:
                    shutil.rmtree(folder)
                except:
                    pass
            sys.stdout.write("Warning: State directories are found, and we will clean all state directories!\n")
        else:
            pass
 
        for i in range(n):
            dirname_mkdir = 'state_' + str(i+1)
            os.mkdir(dirname_mkdir)
            os.chdir(dirname_mkdir)
            shutil.copy('../POSCAR','POSCAR')
            shutil.copy('../POTCAR','POTCAR')
            if os.path.exists('../KPOINTS'):
                shutil.copy('../KPOINTS','KPOINTS')
            else:
                pass
            shutil.copy2('../INCAR.relax1','INCAR.relax1')
            os.chdir('../')

    # update INCAR
    def update_INCAR(self, args):
        """
        Updates the INCAR files in the specified state directories based on the provided thermal input parameters.
    
        This method reads thermal input data from the specified input file and updates the INCAR files 
        for each state directory found in the current working directory. The updates depend on the specified 
        run style (create, NPT, or NVT) and modify parameters such as PSTRESS, TEBEG, and SIGMA accordingly.
    
        Parameters:
        args: An object containing the input file path, lattice information, and run style.
    
        Raises:
        FileNotFoundError: If the specified input file does not exist.
        ValueError: If the state directories do not contain valid state numbers.

        Returns:
        None: This method updates the INCAR files in each state directory.
        """
        folders = glob.glob('state_*')
        thermal_input = Thermal()
        thermal_input.read_thermal_input(args.inputfile, args.lattice)
        pressure_list = thermal_input.P
        temperature_list = thermal_input.T
        state_list = []
        for folder in folders:
            match = re.search(r'state_(\d+)', folder)
            if match:
                state = int(match.group(1))
                state_list.append(state)
        else:
            pass
        #state_list = [re.search(r'state_(\d+)', folder).group for folder in folders]
        #state_list = [int(re.search(r'state_(\d+)', folder).group(1)) for folder in folders]
        
        for state in state_list:
            os.chdir('state_' + str(state))
            if args.runstyle == "create":
                fn = 'INCAR.relax1'
                self.change_INCAR_para('PSTRESS', pressure_list[state-1] * 10, fn)
                shutil.copy2('INCAR.relax1', 'INCAR')
            elif args.runstyle == "NPT":
                fn = 'INCAR.NPT'
                self.change_INCAR_para('PSTRESS', pressure_list[state-1] * 10, fn)
                self.change_INCAR_para('TEBEG', temperature_list[state-1], fn)
                self.change_INCAR_para('SIGMA', temperature_list[state-1] / 11604.525, fn)
                shutil.copy2('INCAR.NPT', 'INCAR')
            elif args.runstyle == "NVT":
                fn = 'INCAR.NVT'
                self.change_INCAR_para('TEBEG', temperature_list[state-1], 'INCAR.NVT')
                self.change_INCAR_para('SIGMA', temperature_list[state-1] / 11604.525, 'INCAR.NVT')
                shutil.copy2('INCAR.NVT', 'INCAR')
            os.chdir('../')

    # change INCAR parameters
    def change_INCAR_para(self, parameter, value, fn):
        """
        Updates the specified parameter in an INCAR file with a new value.
    
        This method reads an INCAR file, searches for a line that starts with the given parameter,
        and replaces its value with the provided value. The updated lines are then written back to the file.
    
        Parameters:
        parameter (str): The name of the parameter to be changed.
        value (any): The new value to assign to the parameter.
        fn (str): The path to the INCAR file to be modified.

        Returns:
        None: This method updates the specified parameter in the INCAR file.
        """
        with open(fn, 'r') as file:
            lines = file.readlines()

        with open(fn, 'w') as file:
            for line in lines:
                if line.strip().startswith(parameter):
                    line = parameter + ' = ' + str(value) + '\n'
                file.write(line)

    # prepare for elastic calculation
    def elastic_calculation_preparation(self, DMAX, args):
        """
        Prepares the necessary files and directories for elastic calculations.
    
        This method organizes the directory structure and copies required input files 
        for each state and destination based on the provided arguments. It handles both 
        general and NPT modes of calculation, ensuring that the appropriate files are 
        copied to the correct locations.
    
        Parameters:
        DMAX (int): The maximum number of destination directories to process.
        args (Namespace): An object containing various parameters including:
            - nvolume (int or None): The number of volumes to process; if None, uses nstate.
            - nstate (int): The number of states to process.
            - nstrain (int): The number of strain configurations to prepare.
            - runstyle (str): The style of the run, affecting the input file naming.
            - mode (str): The calculation mode, which can be 'NPT' or others.
    
        Returns:
        None: 
        - This method prepares directories and copies files for elastic calculations.
        - This method prints messages to the standard output indicating the completion of the preparation.
        - This method provides instructions for running the calculations.
        """
        if args.nvolume == None:
            n = args.nstate
        else:
            n = args.nvolume
        
        for i in range(n):
            dirname_v = 'state_' + str(i+1) 
            os.chdir(dirname_v)
            shutil.copy2('../INCAR.'+args.runstyle, 'INCAR.'+args.runstyle)
            os.chdir('../')
        self.update_INCAR(args)

        for i in range(n):
            dirname_v = 'state_' + str(i+1) 
            os.chdir('state_' + str(i+1))
            if args.mode != 'NPT':
                for j in range(1, DMAX + 1):
                    if j < 10:
                        dirname = 'Dst_0' + str(j)
                    else:
                        dirname = 'Dst_' + str(j)
                    if os.path.exists(dirname):
                        os.chdir(dirname)
                        for k in range(args.nstrain):
                            dirname_k = str(k+1)
                            os.chdir(dirname_k)
                            if args.runstyle == 'static' and os.path.exists('CONTCAR'):
                                shutil.copy2('CONTCAR','POSCAR')
                            else:
                                shutil.copy2('POSCAR','POSCAR_s')
                            shutil.copy2('../../../POTCAR', 'POTCAR')
                            INCAR_file = '../../INCAR.'+args.runstyle
                            shutil.copy2(INCAR_file, 'INCAR')
                            if os.path.exists('../../../KPOINTS'):
                                shutil.copy2('../../../KPOINTS', 'KPOINTS')
                            os.chdir('../')                        
                        os.chdir('../')                    
                os.chdir('../')
            else:
                os.chdir('NPT')
                shutil.copy2('../../POTCAR', 'POTCAR')
                shutil.copy2('../INCAR.NPT', 'INCAR')
                if os.path.exists('../../KPOINTS'):
                    shutil.copy2('../../KPOINTS', 'KPOINTS')
                os.chdir('../../')
        if args.mode != 'NPT':
            sys.stdout.write('The preparation for elastic calculation is finished!\n')
            sys.stdout.write('Please run the calculation at state_*/Dst_*/{'+str(1)+'..'+str(args.nstrain)+'}.\n')
        else:
            sys.stdout.write('The preparation for elastic calculation is finished!\n')
            sys.stdout.write('Please run the calculation at state_*/NPT.\n')

    # isothermal elastic constants to isentropic elastic constants
    def trans_isentroestc(self, C_S, thermal_input, i):
        """
        Transforms the isentropic state of a system based on the provided thermal input and material properties.
    
        Parameters:
        C_S (object): An object containing the material's property matrix (C_matrix) and volume (V).
        thermal_input (object): An object containing thermal properties such as temperature (T), specific heat (Cp),
                                and coefficients (a1, a2, a3) for the calculations.
        i (int): The index to access specific thermal properties from the thermal_input object.
    
        The function calculates the coefficients b1 to b6 based on the material property matrix and the coefficients
        from the thermal input. It then updates the material property matrix C based on the temperature and specific
        heat, ensuring that the calculations respect the physical constraints of the system. If the temperature is zero
        or specific heat is non-positive, the matrix is preserved without changes.
        
        Returns:
        None: This method updates the material property matrix based on the thermal
              input and physical constraints of the system and the results is stored in
              object C_S.

        Note:
        The transformation is based on the following equations:
        - C' = C + T * V * b * b / c_n * 1E-39
        - c_n = Cp / Na - T * V * 1E-21 * (C[0,0] * a1 * a1 + 2 * C[0,1] * a1 * a2 + 2 * C[0,2] * a1 * a3 
                                          + C[1,1] * a2 * a2 + 2 * C[1,2] * a2 * a3 + C[2,2] * a3 * a3 )

        The coefficients b1 to b6 are calculated as follows:
        - b1 = -1E9 * (C[0,0] * a1 + C[0,1] * a2 + C[0,2] * a3)
        - b2 = -1E9 * (C[0,1] * a1 + C[1,1] * a2 + C[1,2] * a3)
        - b3 = -1E9 * (C[0,2] * a1 + C[1,2] * a2 + C[2,2] * a3)
        - b4 = -1E9 * (C[0,3] * a1 + C[1,3] * a2 + C[2,3] * a3)
        - b5 = -1E9 * (C[0,4] * a1 + C[1,4] * a2 + C[2,4] * a3)
        - b6 = -1E9 * (C[0,5] * a1 + C[1,5] * a2 + C[2,5] * a3)

        The transformation is applied to the material property matrix C based on the temperature and specific heat.
        """
        Na = Basic_para().Na
        C = C_S.C_matrix
        V = C_S.V
        T = thermal_input.T[i]
        Cp = thermal_input.Cp[i]
        a1 = thermal_input.a[i]
        a2 = thermal_input.b[i]
        a3 = thermal_input.c[i]
 
        b1 = -1E9 * (C[0,0] * a1 + C[0,1] * a2 + C[0,2] * a3) 
        b2 = -1E9 * (C[0,1] * a1 + C[1,1] * a2 + C[1,2] * a3)  
        b3 = -1E9 * (C[0,2] * a1 + C[1,2] * a2 + C[2,2] * a3) 
        b4 = -1E9 * (C[0,3] * a1 + C[1,3] * a2 + C[2,3] * a3) 
        b5 = -1E9 * (C[0,4] * a1 + C[1,4] * a2 + C[2,4] * a3) 
        b6 = -1E9 * (C[0,5] * a1 + C[1,5] * a2 + C[2,5] * a3)       
        c_n = Cp / Na - T * V * 1E-21 * (C[0,0] * a1 * a1 + 2 * C[0,1] * a1 * a2 + 2 * C[0,2] * a1 * a3 
                                       + C[1,1] * a2 * a2 + 2 * C[1,2] * a2 * a3 + C[2,2] * a3 * a3 )
        if T == 0 or Cp <= 0:
            C[0,0] = C[0,0] 
            C[0,1] = C[1,0]  = C[0,1] 
            C[0,2] = C[2,0]  = C[0,2] 
            C[0,3] = C[3,0]  = C[0,3] 
            C[0,4] = C[4,0]  = C[0,4] 
            C[0,5] = C[5,0]  = C[0,5] 
 
            C[1,1] = C[1,1] 
            C[1,2] = C[2,1]  = C[1,2] 
            C[1,3] = C[3,1]  = C[1,3] 
            C[1,4] = C[4,1]  = C[1,4] 
            C[1,5] = C[5,1]  = C[1,5] 
 
            C[2,2] = C[2,2] 
            C[2,3] = C[3,2]  = C[2,3] 
            C[2,4] = C[4,2]  = C[2,4] 
            C[2,5] = C[5,2]  = C[2,5] 
 
            C[3,3] = C[3,3] 
            C[3,4] = C[4,3]  = C[3,4] 
            C[3,5] = C[5,3]  = C[3,5] 
 
            C[4,4] = C[4,4] 
            C[4,5] = C[5,4]  = C[4,5] 
 
            C[5,5] = C[5,5] 
        else:
            C[0,0] = C[0,0] + T * V * b1 * b1 / c_n * 1E-39
 
            C[0,1] = C[1,0] = C[0,1] + T * V * b1 * b2 / c_n * 1E-39
            C[0,2] = C[2,0] = C[0,2] + T * V * b1 * b3 / c_n * 1E-39
            C[0,3] = C[3,0] = C[0,3] + T * V * b1 * b4 / c_n * 1E-39
            C[0,4] = C[4,0] = C[0,4] + T * V * b1 * b5 / c_n * 1E-39
            C[0,5] = C[5,0] = C[0,5] + T * V * b1 * b6 / c_n * 1E-39
 
            C[1,1] = C[1,1] + T * V * b1 * b2 / c_n  * 1E-39
            C[1,2] = C[2,1] = C[1,2] + T * V * b2 * b3 / c_n * 1E-39
            C[1,3] = C[3,1] = C[1,3] + T * V * b2 * b4 / c_n * 1E-39
            C[1,4] = C[4,1] = C[1,4] + T * V * b2 * b5 / c_n * 1E-39
            C[1,5] = C[5,1] = C[1,5] + T * V * b2 * b6 / c_n * 1E-39
 
            C[2,2] = C[2,2] + T * V * b3 * b3 / c_n * 1E-39
            C[2,3] = C[3,2] = C[2,3] + T * V * b3 * b4 / c_n * 1E-39
            C[2,4] = C[4,2] = C[2,4] + T * V * b3 * b5 / c_n * 1E-39
            C[2,5] = C[5,2] = C[2,5] + T * V * b3 * b6 / c_n * 1E-39
 
            C[3,3] = C[3,3] + T * V * b4 * b4 / c_n * 1E-39
            C[3,4] = C[4,3] = C[3,4] + T * V * b4 * b5 / c_n * 1E-39
            C[3,5] = C[5,3] = C[3,5] + T * V * b4 * b6 / c_n * 1E-39
 
            C[4,4] = C[4,4] + T * V * b5 * b5 / c_n * 1E-39
            C[4,5] = C[5,4] = C[4,5] + T * V * b5 * b6 / c_n * 1E-39
 
            C[5,5] = C[5,5] + T * V * b5 * b6 / c_n * 1E-39
    
    # summary for elastic results at different state
    def summary_results(self,args):
        """
        Summarizes the results of elasticity calculations based on the provided arguments.
    
        This method determines whether to operate in 'state' or 'volume' mode based on the input arguments.
        It calculates the elasticity tensors for different states or volumes, interpolates the results if necessary,
        and writes the output to files. The method handles exceptions for missing directories and ensures that
        default fitting orders are applied when required.
    
        Parameters:
        args: An object containing various parameters including nstate, nvolume, method, forder, mode, inputfile, lattice, and natom.
    
        Returns:
        None: The results are written directly to output files.
        
        Raises:
        FileNotFoundError: If the specified input file does not exist.
        ValueError: If the specified fitting order is less than the required value for the method.
        
        Notes:
        - The method assumes the existence of certain classes and methods such as Elasticity, Thermal, and write_output.
        - The output files are named 'Elasticity_T.dat' and 'Elasticity_S.dat'.
        - The method prints status messages to the standard output and error streams.
        """
        C_0K = {}
        C_T  = {}
        C_S  = {}
        if args.nvolume == None:
            n = args.nstate
            mode = "state"
        else:
            n = args.nvolume
            mode = "volume"

        if args.forder == None or (args.forder < 2 and args.method == 'Energy') or (args.forder < 1 and args.method == 'Stress'):
            if args.method == 'Energy':
                args.forder = 3
            else:
                args.forder = 2
            if args.mode != 'NPT':
                sys.stdout.write('Using default fitting order = %d\n' % args.forder)
        
        calculated_list = []

        for i in range(n):
            dirname = 'state_' + str(i+1)
            C_0K[i] = Elasticity()
            try:
                os.chdir(dirname) 
                sys.stdout.write('Start to get results of ' + dirname + '\n')
            except:
                sys.stderr.write(dirname + "does not exist, and this state is skipped!\n")
                continue
            calculated_list.append(i)
            self.get_one_state_results(args, C_0K[i], i)
            os.chdir('../')

        thermal_input = Thermal()

        thermal_input.read_thermal_input(args.inputfile, args.lattice)
        volume_input = thermal_input.V * args.natom

        if mode == "volume":
            vmax = max(volume_input)
            vmin = min(volume_input)
            volume_calc = np.linspace(vmin, vmax, args.nvolume)
            volume_calc = np.array([volume_calc[i] for i in calculated_list])
        elif mode == "state":
            volume_calc = volume_input
 
        if mode == "state" or (mode == "volume" and n == 1):
            for i in range(n):
                C_T[i] = Elasticity()
                C_T[i].C_matrix = C_0K[i].C_matrix
        elif (mode == "volume") and len(calculated_list) >= 4:
            cs = {}
            for i in range(1, 7):
                for j in range(1, 7):
                    C_ij = [C_0K[k].C(i,j) for k in calculated_list]
                    cs[i,j] = interpolate.CubicSpline(volume_calc, C_ij)
            for i in range(len(volume_input)):
                C_T[i] = Elasticity()
                C_temp = np.zeros((6,6))
                for j in range(1, 7):
                    for k in range(1, 7):
                        C_temp[j-1, k-1] = cs[j,k](volume_input[i])
                C_T[i].C_matrix = C_temp
 
        fn_ET = 'Elasticity_T.dat'
        fn_ES = 'Elasticity_S.dat'
        if  args.mode != 'cold':
            write_output().filehead(args.lattice, fn_ET)
            write_output().filehead(args.lattice, fn_ES)
            for i in range(len(volume_input)):
                if mode == "state" and i not in calculated_list:
                    continue
                C_S[i] = Elasticity()
                C_S[i].T = C_T[i].T = thermal_input.T[i]
                C_S[i].V = C_T[i].V = thermal_input.V[i]
                C_S[i].P = C_T[i].P = thermal_input.P[i]
                C_S[i].rho = C_T[i].rho = thermal_input.rho[i]
                C_S[i].C_matrix = C_T[i].C_matrix
                C_T[i].cal_properties(args)
                C_S[i].cal_properties(args)
                write_output().Cinf(args.lattice, fn_ET, C_T[i])
                self.trans_isentroestc(C_S[i], thermal_input, i)
                write_output().Cinf(args.lattice, fn_ES, C_S[i])
        else:
            write_output().filehead(args.lattice, fn_ET)
            for i in range(len(volume_input)):
                if mode == "state" and i not in calculated_list:
                    continue
                C_T[i].T = thermal_input.T[i]
                C_T[i].V = thermal_input.V[i]
                C_T[i].P = thermal_input.P[i]
                C_T[i].rho = thermal_input.rho[i]
                C_T[i].cal_properties(args)
                write_output().Cinf(args.lattice, fn_ET, C_T[i])
        if args.mode != 'cold':
            sys.stdout.write('Isothermal elasticity have been saved to Elasticity_T.dat.\n')
            sys.stdout.write('Isentropic elasticity have been saved to Elasticity_S.dat.\n')
        else:
            sys.stdout.write('Isothermal elasticity have been saved to Elasticity_T.dat.\n')

    # get cell matrix
    def get_cell_matrix(self, input='POSCAR'):
        """
        Extracts the cell matrix from a POSCAR file.
    
        This function reads a POSCAR file and retrieves the original tensor matrix 
        defined by the first three vectors in the file. It assumes that the vectors 
        are located on lines 3, 4, and 5 of the file. The resulting matrix is returned 
        as a NumPy matrix.
    
        Parameters:
        input (str): The path to the POSCAR file. Defaults to 'POSCAR'.
    
        Returns:
        np.matrix: A 3x3 matrix representing the cell vectors.
        """
        #Original Tensor matrix from input POSCAR
        lines = open(input, 'r').readlines()
        cnt = 1
        for line in lines:
            if cnt == 3:
                res = line.split()
                xx = float(res[0])
                xy = float(res[1])
                xz = float(res[2])
            elif cnt == 4:
                res = line.split()
                yx = float(res[0])
                yy = float(res[1])
                yz = float(res[2])
            elif cnt == 5:
                res = line.split()
                zx = float(res[0])
                zy = float(res[1])
                zz = float(res[2])
            else:
                pass
            cnt += 1
 
        cellm = np.matrix([
            [xx, xy, xz],
            [yx, yy, yz],
            [zx, zy, zz]
        ])
 
        return cellm

    # get elastic properties for one directory
    def get_one_state_results(self,args, C, i):
        """
        Calculates and retrieves the results for a specific state based on the provided method (Energy or Stress).
        This function handles different modes of operation and extracts relevant physical properties such as pressure,
        temperature, and compliance matrix. It utilizes various classes to read input data and perform calculations.
        
        Parameters:
        args: An object containing various input parameters including lattice, cij_order, fit order, method, 
                number of tasks, mode, MLFF, and input file.
        C: An object to store the calculated results including volume, pressure, temperature, and compliance matrix.
        i: An index used to access specific thermal properties from the Thermal class.
        
        Returns:
        None: The results are stored directly in the provided object C.
        """
        LC = args.lattice
        cij_order = args.cij_order
        fitorder = args.forder
        method = args.method
        num_tasks = args.nstrain
        mode = args.mode
        MLFF = args.MLFF
        thermal = Thermal()
        fn_path = os.path.join('..', args.inputfile)
        thermal.read_thermal_input(fn_path, LC)
        if method == 'Energy':
            Energy_Strain().get_results_energy(LC, num_tasks)
            C0 = Energy_Strain().fit_res_energy(LC, method, cij_order, fitorder, args.plt)
            Cf = Energy_Strain().format_Cij_energy(C0, LC, fitorder)
            C.V = self.get_volume('POSCAR', 'cell')  # ground state volume
            if args.inputfile != None and os.path.exists('OUTCAR'):
                C.P = self.extract_pressure('OUTCAR')
            else:
                C.P = thermal.P[i]
            C.T = thermal.T[i]
            C.C_matrix = Cf
            self.get_all_mod(C, units='eV', use_symmetry = True, method = 'Energy')
        elif method == 'Stress':
            if mode in ['cold', 'QSA', 'NVT']:
                Stress_Strain().get_results_stress(LC, mode, MLFF)
                C0 = Stress_Strain().fit_res_stress(LC, cij_order, fitorder, args.plt)
                Cf = Stress_Strain().format_Cij_stress(C0, LC)
                C.C_matrix = Cf
            elif mode == 'NPT':
                os.chdir('NPT')
                self.extract_stress(fin='OUTCAR', mode='NPT', MLFF=MLFF)
                NPT().extract_structure()
                self.extract_pressure_MD('OUTCAR')
                self.extract_temperature_MD('OUTCAR')
                self.extract_volume_MD('OUTCAR')
                P_input = thermal.P[i]
                maxratio = 1+args.maxstrain
                #end = len(np.loadtxt("stress.out")[:,0])-1
                start,end = NPT().adapative_sampling_algorithm(limit = maxratio)
                NPT().extract_strain(start,end)
                C.C_matrix = NPT().solve_cij(LC, start, end)
                os.chdir('../')
            self.get_all_mod(C, units='kbar', use_symmetry = True, method = 'Stress')

    # read stress information
    def read_stress(self, fin='stress.out',mode='cold'):
        """
        Reads stress data from a specified output file and returns a 3x3 stress tensor.
    
        Parameters:
        fin (str): The filename to read stress data from. Default is 'stress.out'.
        mode (str): The mode of reading stress data. Options are 'cold', 'QSA', or 'NVT'.
                    - 'cold' or 'QSA': Reads the last line or single line of the file.
                    - 'NVT': Averages the stress values across all lines in the file.
    
        Returns:
        numpy.ndarray: A 3x3 numpy array representing the stress tensor, negated.
        
        Raises:
        FileNotFoundError: If the specified file does not exist.
        """
        if os.path.exists(fin):
            if mode == 'cold' or mode == 'QSA':
                if len(np.loadtxt(fin).shape) == 1:
                    ss = np.loadtxt(fin)
                elif len(np.loadtxt(fin).shape) == 2:
                    ss = np.loadtxt(fin)[-1,:]
                stress = np.zeros((3, 3))
                stress[0, 0] = ss[0]
                stress[1, 1] = ss[1]
                stress[2, 2] = ss[2]
                stress[0, 1] = ss[3]
                stress[1, 2] = ss[4]
                stress[0, 2] = ss[5]
                for i in range(3):
                    for j in range(3):
                        stress[j, i] = stress[i, j]
                stress = -stress
            elif mode == 'NVT':
                ss = np.loadtxt(fin)
                stress = np.zeros((3, 3))
                stress[0, 0] = np.average(ss[:,0])
                stress[1, 1] = np.average(ss[:,1])
                stress[2, 2] = np.average(ss[:,2])
                stress[0, 1] = np.average(ss[:,3])
                stress[1, 2] = np.average(ss[:,4])
                stress[0, 2] = np.average(ss[:,5])
                for i in range(3):
                    for j in range(3):
                        stress[j, i] = stress[i, j]
                stress = -stress
        return stress

    # analyze anisotropy information
    def anisotropy(self, args, fn):
        """
        Analyzes the anisotropy of materials based on the provided modulus and file input.
    
        Parameters:
        args (Namespace): Command line arguments containing modulus type and plotting preferences.
        fn (str): The filename containing elasticity data, expected to be either 'Elasticity_T.dat' or 'Elasticity_S.dat'.
    
        This method reads the elasticity data from the specified file, processes it according to the modulus type,
        and generates corresponding anisotropy figures and data files. It supports different modulus types including
        bulk modulus (B), shear modulus (G), Young's modulus (E), Poisson's ratio (nu), sound speed (sound), 
        and gif generation.
    
        Depending on the modulus type, it calls appropriate methods to calculate the anisotropy and save the results.
        It also handles error cases for unsupported modulus types and provides feedback on the output files generated.
        """
        with open(fn, 'r') as file:
            lines = file.readlines()
        n_lines = len(lines)- 2
        C = {}
        #if args.amode == 'plot':
        #    if args.modulus not in ['B','G','E','v','sound']:
        #        sys.tderr.write("This modulus is not correct, please choose one: B, G, E, v, sound!\n")
        #        os.sys.exit(1)
        if args.modulus in ['B','G','E','nu','sound']:
            if fn == 'Elasticity_T.dat':
                sys.stdout.write("Using the Elasticity_T.dat file to analyze anisotropy of "+ args.modulus +"\n")
            elif fn == 'Elasticity_S.dat':
                sys.stdout.write("Using the Elasticity_S.dat file to analyze anisotropy of "+ args.modulus +"\n")
            else:
                sys.stdout.write("Using the "+ fn +" file to analyze anisotropy of "+ args.modulus +"\n")
            for i in range(n_lines):
                C[i] = Elasticity()
                C[i].read_output(fn, args, i)
                if fn == 'Elasticity_T.dat':
                    state = 'isothermal'
                elif fn == 'Elasticity_S.dat':
                    state = 'isentropic'
                else:
                    state = 'other'
                
                if args.modulus in ['B','G','E','nu']:
                    Anisotropy().calc_modulus_3D(C[i], args, state, i)
                elif args.modulus in ['sound']:
                    Anisotropy().calc_sound_3D(C[i], args, state, i)
            if fn == 'Elasticity_T.dat':
                if args.plt in ['png', 'eps']:
                    sys.stdout.write("All anisotropy figures and data files of isothermal "+ args.modulus +" have been saved at figures/anisotropy \n")
                else:
                    sys.stdout.write("All data files of isothermal anisotropy "+ args.modulus +" have been saved at figures/anisotropy \n")
            elif fn == 'Elasticity_S.dat':
                if args.plt in ['png', 'eps']:
                    sys.stdout.write("All anisotropy figures and data files of isentropic "+ args.modulus +" have been saved at figures/anisotropy \n")
                else:
                    sys.stdout.write("All data files of isentropic anisotropy "+ args.modulus +" have been saved at figures/anisotropy \n")
            else:
                if args.plt in ['png', 'eps']:
                    sys.stdout.write("All anisotropy figures and data files of "+ args.modulus +" have been saved at figures/anisotropy \n")
                else:
                    sys.stdout.write("All data files of anisotropy "+ args.modulus +" have been saved at figures/anisotropy \n")

        elif args.modulus == 'gif':
            if fn == 'Elasticity_T.dat':
                sys.stdout.write("Using the isothermal anisotropy png files at figures/anisotropy/ to make gif\n")
                Anisotropy().png_gif("isothermal")
                sys.stdout.write("All gif files of isothermal anisotropy have been saved at figures/anisotropy/ \n")
            elif fn == 'Elasticity_S.dat':
                sys.stdout.write("Using the isentropic anisotropy png files at figures/anisotropy/ to make gif\n")
                Anisotropy().png_gif("isentropic")
                sys.stdout.write("All gif files of isentropic anisotropy have been saved at figures/anisotropy/ \n")
            else:
                sys.stdout.write("Using the anisotropy png files at figures/anisotropy/ to make gif\n")
                Anisotropy().png_gif("other")
                sys.stdout.write("All gif files of anisotropy have been saved at figures/anisotropy/ \n")
        else:
            sys.stderr.write("Error: The modulus is not correct, please choose one: B, G, E, v, sound, gif!\n")
            os.sys.exit(1)

    # model elastic properties at WT and WP
    def SAM_PT(self, args, fn):
        """
        SAM_PT processes input data from specified files to perform elasticity calculations.
        It reads the contents of the file, determines the state based on the filename, 
        and initializes Elasticity objects for each line of data. The function checks 
        if the reference temperature is present in the data and calls the model_elasticity 
        method of the SAM class to perform the calculations. It also provides feedback 
        on the output file format and location.
        
        Parameters:
        args: An object containing various parameters including reference temperature 
            and output format.
        fn: A string representing the filename to be processed, which can be 
            'Elasticity_T.dat', 'Elasticity_S.dat', or other files.
        
        Raises:
        SystemExit: If the reference temperature is not found in the input file.
        """
        with open(fn, 'r') as file:
            lines = file.readlines()
        n_lines = len(lines)-2
        C = {}
        if fn == 'Elasticity_T.dat':
            sys.stdout.write("Using the Elasticity_T.dat file as input for SAM\n")
            state = 'isothermal'
        elif fn == 'Elasticity_S.dat':
            sys.stdout.write("Using the Elasticity_S.dat file as input for SAM\n")
            state = 'isentropic'
        else:
            sys.stdout.write("Using the "+ fn +" file as input for SAM\n")
            state = 'other'
        for i in range(n_lines):
            C[i] = Elasticity()
            C[i].read_output(fn, args, i)
            #C[i].cal_properties(args)
        if args.T_ref not in [C[i].T for i in range(n_lines)]:
            sys.stderr.write('Error: The reference temperature can not be found in ' + fn +' !\n')
            os.sys.exit(1)
        SAM().model_elasticity(C, args, state)
        if args.plt in ['png', 'eps']:
            sys.stdout.write("All figures and data files of SAM have been saved at figures/model \n")
        else:
            sys.stdout.write("All data files of SAM have been saved at figures/model \n")
        
    """
    # get the directory list
    def get_dst_dirlist(self, Target=os.getcwd()):
        CURDIR = os.getcwd()
        os.chdir(Target)
        print(os.getcwd())
        #try: !!修改检查和清理文件夹的命令
        #    os.system('ls -d state_*/Dst_?? > Dstdirs')
        #except FileNotFoundError:
        #    pass
        lines = open('Dstdirs').readlines()
        n_lines = len(lines)
        dirlist = []
        for i in range(n_lines):
            dirlist.append(lines[i].split()[0])
        os.chdir(CURDIR)
        return dirlist
    
    # clean files
    def clean_files(self): #!!clean命令需要修改
        DirList = self.get_dst_dirlist()
        for dirtmp in DirList:
            if os.path.exists(dirtmp):
                sys.stdout.write('Cleaning DIR-%s ...\n' % dirtmp)
                shutil.rmtree(dirtmp)
    """

#    def sortlist(lst1, lst2): 
#        temp = copy.copy(lst1)
#        lst3 = []
#        lst4 = []
#        temp.sort()
#        for i in range(len(lst1)):
#            lst3.append(lst1[lst1.index(temp[i])])
#            lst4.append(lst2[lst1.index(temp[i])])
#        return lst3, lst4

class Strain:
    """
    This module defines the `Strain` class, which provides methods for generating and applying distortions to material structures based on specified strain conditions. 

    The `Strain` class includes functionalities to:
    - Generate a dictionary of distortion parameters for a given computational code (e.g., VASP).
    - Retrieve a list of lag strain values based on loading conditions and methods (Energy or Stress).
    - Create strain vectors within a specified range.
    - Apply strains to a material structure and generate distorted structures.
    - Load distortion information from a file.
    - Set and replace deformation matrices in POSCAR files.

    Each method is documented with parameters, return types, and potential exceptions raised, ensuring clarity for users and maintainers of the code.
    """

    # list of distortion
    def list(self, code='VASP'):
        """
        Generates a distortion dictionary for the specified code.
    
        Parameters:
        code (str): The code for which to generate the distortion dictionary. 
            Default is 'VASP'. Currently, only 'VASP' is supported.
    
        Returns:
        dict: A dictionary where keys are distortion identifiers (str) and 
            values are lists of distortion parameters (list of float).
        """
        if code in ['VASP']:
            distort_dic = {
                '01': [1., 1., 1., 0., 0., 0.],
                '02': [1., 0., 0., 0., 0., 0.],
                '03': [0., 1., 0., 0., 0., 0.],
                '04': [0., 0., 1., 0., 0., 0.],
                '05': [0., 0., 0., 2., 0., 0.],
                '06': [0., 0., 0., 0., 2., 0.],
                '07': [0., 0., 0., 0., 0., 2.],
                '08': [1., 1., 0., 0., 0., 0.],
                '09': [1., 0., 1., 0., 0., 0.],
                '10': [1., 0., 0., 2., 0., 0.],
                '11': [1., 0., 0., 0., 2., 0.],
                '12': [1., 0., 0., 0., 0., 2.],
                '13': [0., 1., 1., 0., 0., 0.],
                '14': [0., 1., 0., 2., 0., 0.],
                '15': [0., 1., 0., 0., 2., 0.],
                '16': [0., 1., 0., 0., 0., 2.],
                '17': [0., 0., 1., 2., 0., 0.],
                '18': [0., 0., 1., 0., 2., 0.],
                '19': [0., 0., 1., 0., 0., 2.],
                '20': [0., 0., 0., 2., 2., 0.],
                '21': [0., 0., 0., 2., 0., 2.],
                '22': [0., 0., 0., 0., 2., 2.],
                '23': [0., 0., 0., 2., 2., 2.],
                '24': [-1., .5, .5, 0., 0., 0.],
                '25': [.5, -1., .5, 0., 0., 0.],
                '26': [.5, .5, -1., 0., 0., 0.],
                '27': [1., -1., 0., 0., 0., 0.],
                '28': [1., -1., 0., 2., 0., 0.],
                '29': [1., -1., 0., 0., 0., 2.],
                '30': [1., 0., -1., 0., 2., 0.],
                '31': [0., 1., -1., 0., 0., 2.],
                '32': [1., 1., -1., 2., 2., 2.],
                '33': [1., 0., 0., 2., 2., 0.],
                '34': [0., 1., 0., 2., 2., 0.],
                '35': [1., 1., 0., 2., 2., 0.],
                '36': [1., 1., 0., 2., 0., 0.],
                '37': [1., 1., -1., 0., 0., 0.],
                '38': [1., 1., 1., -2., -2., -2.],
                '39': [1., 2., 3., 4., 5., 6.],
                '40': [-2., 1., 4., -3., 6., -5.],
                '41': [3., -5., -1., 6., 2., -4.],
                '42': [-4., -6., 5., 1., -3., 2.],
                '43': [5., 4., 6., -2., -1., -3.],
                '44': [-6., 3., -2., 5., -4., 1.],
                '45': [1., 0., 0., 0., 0., 0.],
                '46': [0., 1., 0., 0., 0., 0.],
                '47': [0., 0., 1., 0., 0., 0.],
                '48': [0., 0., 0., 1., 0., 0.],
                '49': [0., 0., 0., 0., 1., 0.],
                '50': [0., 0., 0., 0., 0., 1.],
                '51': [1., 0., 0., 0., 0., 1.],
                '52': [0., 0., 1., 0., 0., 0.],
                '53': [1., 0., 0., 1., 0., 0.],
                '54': [0., .5, 1., 1., 0., 0.],
                '55': [1., 1., 0., 0., 0., 1.],
                '56': [0., 0., 1., 0., 0., 1.],
                '57': [0., 1., 0., 1., 0., 0.],
            #    '58': [0., 0., 1., 0., 0., 1.],
            #    '59': [-.5, .5, 0., 0., 1., 0.],
            #    '60': [.5, .5, 0., 1., 0., 0.],
                '58': [1., 0., 0., 1., 0., 0.],
                '59': [0., 1., 0., 0., 1., 0.],
                '60': [0., 0., 1., 0., 0., 1.],
                '61': [1., 0., 0., 1., 0., 0.],
                '62': [0., 1., 0., 0., 1., 0.],
            }
        return distort_dic

    # get the distortion list for different lattice
    def get_list(self, LC='C', method='Energy', order=2):
        """
        Retrieves a list of lag strain values based on the specified loading condition (LC),
        method (Energy or Stress), and order (currently only order 2 is implemented).
    
        Parameters:
        LC (str): The loading condition, which can be one of the following:
                  'C', 'H', 'RI', 'RII', 'TI', 'TII', 'O', 'M', 'N'.
        method (str): The method to use for retrieving the strain list, either 'Energy' or 'Stress'.
        order (int): The order of the method, currently only order 2 is supported.
    
        Returns:
        list: A list of lag strain values corresponding to the specified parameters.
    
        Raises:
        SystemExit: If order is not 2, an error message is written to stderr and the program exits.
        """
        if (order == 2):
            if (method == 'Energy'):
                if (LC == 'C'):
                    Lag_strain_list = ['01', '08', '23']
                if (LC == 'H'):
                    Lag_strain_list = ['01', '26', '04', '03', '17']
                if (LC == 'RI'):
                    Lag_strain_list = ['01', '08', '04', '02', '05', '10']
                if (LC == 'RII'):
                    Lag_strain_list = [
                        '01', '08', '04', '02', '05', '10', '11']
                if (LC == 'TI'):
                    Lag_strain_list = ['01', '26', '27', '04', '05', '07']
                if (LC == 'TII'):
                    Lag_strain_list = [
                        '01', '26', '27', '29', '04', '05', '07']
                if (LC == 'O'):
                    Lag_strain_list = [
                        '01', '26', '25', '27', '03', '04', '05', '06', '07']
                if (LC == 'M'):
                    Lag_strain_list = [
                        '01', '25', '24', '19', '29', '27', '20', '12', '03', '04',
                        '05', '06', '07']
                if (LC == 'N'):
                    Lag_strain_list = [
                        '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                        '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                        '22']
            elif (method == 'Stress'):
                if (LC == 'C'):
                    Lag_strain_list = ['51']
                if (LC == 'H'):
                    Lag_strain_list = ['52', '53']
                if (LC == 'TI'):
                    Lag_strain_list = ['54', '55']
                if (LC == 'TII'):
                    Lag_strain_list = ['54', '55']
                if (LC == 'RI'):
                    Lag_strain_list = ['56', '57']
                if (LC == 'RII'):
                    Lag_strain_list = ['56', '57']
                if (LC == 'O'):
                    Lag_strain_list = ['58', '59', '60']
                if (LC == 'M'):
                    #Lag_strain_list = ['47', '50', '61', '62']
                    Lag_strain_list = ['47', '50', '58', '59']
                if (LC == 'N'):
                    Lag_strain_list = ['45', '46', '47', '48', '49', '50']
        elif (order == 3):
            sys.stderr.write('Error: Not implemented yet. \n')
            os.sys.exit(1)
        return Lag_strain_list

    # set the strain range and strain number
    def range(self, maxstrain, ndata, method='Energy'):
        """
        Generate strain vectors within the range of [-maxstrain, maxstrain].
    
        Parameters:
        maxstrain (float): The maximum strain value.
        ndata (int): The number of data points to generate.
        method (str): The method to use for generating strain vectors. 
                      Options are 'Energy' for a symmetric range and 
                      'Stress' for a range excluding zero.
    
        Returns:
        list: A list of strain values based on the specified method.

        Raises:
        ValueError: If the method is not 'Energy' or 'Stress', a ValueError is raised.
        """
        # strain vectors in range [-maxstrain, maxstrain] with ndata point
        if method == 'Energy':
            etas = np.linspace(-maxstrain, maxstrain, ndata)
        elif method == 'Stress':
            etas = np.linspace(-maxstrain, maxstrain, ndata+1)
            etas = [xi for xi in etas if abs(xi) != 0]
        return etas

    # apply the strains to the structure
    def apply(self,maxstrain, ndata, LC='C', method='Energy', order=2):
        """
        Applies a specified strain to a material and generates the corresponding distorted structures.
    
        Parameters:
        maxstrain (float): The maximum strain to be applied.
        ndata (int): The number of strain data points to generate.
        LC (str, optional): The type of loading condition. Defaults to 'C'.
        method (str, optional): The method used for strain generation. Defaults to 'Energy'.
        order (int, optional): The order of the strain. Defaults to 2.

        Returns:
        None: The method generates distorted structures and writes the applied strain types and values to a file.

        Raises:
        FileNotFoundError: If the 'POSCAR' file is not found.

        This method performs the following steps:
        1. Retrieves the original cell matrix from the 'POSCAR' file.
        2. Calculates the deformation strain range based on the provided parameters.
        3. Generates a list of strain types based on the loading condition and method.
        4. Writes the applied strain types and values to a file for post-processing.
        5. Creates directories for each strain type and generates distorted structures by modifying the cell matrix.
        6. Copies necessary files for VASP or CESSP simulations.
    
        Note:
        The method assumes the existence of certain helper functions and external files (e.g., 'POSCAR').
        """

        # get the original cell matrix
        cell_matrix0 = Common().get_cell_matrix('POSCAR')
        I_matrix = np.eye(3)
 
        # get the deformation strain range
        et_range = self.range(maxstrain, ndata, method)
        ndata = len(et_range)
 
        # Get strain list by inputs
        strain_list = self.get_list(LC, method, order)
 
        # write some information for postprocessing.
        fw1_name = 'Distort_info.dat'
        fw1 = open(fw1_name, 'w+')
        fw1.write("Applied Strain Types:\n")
        for i in strain_list:
            fw1.write("\'%s\'\t" % i)
            fw1.flush()
        fw1.write('\n%d strains applied.\n' % ndata)
        for i in range(ndata):
            fw1.write('%f\n' % et_range[i])
            fw1.flush()
        fw1.close()
 
        # make dirs for each strain lists
        for key in strain_list:
            dirname_tmp = 'Dst_' + key
            if os.path.exists(dirname_tmp): #!!重复清理
                shutil.rmtree(dirname_tmp)
            os.mkdir(dirname_tmp)
            # enter Dis_* dir
            os.chdir(dirname_tmp)
            fw_tmp_name = dirname_tmp + '.dat'
 
            # write strain data points information
            fw_tmp = open(fw_tmp_name, 'w+')
            for i in range(1,ndata+1):
                fw_tmp.write('%f\n' % et_range[i-1])
                fw_tmp.flush()
            fw_tmp.close()
 
            # copy some files to make vasp or cessp run properly
            shutil.copy2('../POSCAR', 'POSCAR_o')
 
            # POSCAR generating
            distort_dic = self.list()
            elist_key = distort_dic[key]
            TASK_LIST = self.get_task_nlist(ndata)
            for i in range(1,ndata+1):
                task_name = TASK_LIST[i-1]
                os.mkdir(task_name)
                os.chdir(task_name)
                shutil.copy2('../POSCAR_o', 'POSCAR_o')
                deform_matrix_i = I_matrix + \
                    et_range[i-1] * self.set_deform_matrix(elist_key)
                cell_matrix_i = np.dot(cell_matrix0, deform_matrix_i)
                self.replace_POSCAR_cell(cell_matrix_i, 'POSCAR_o', 'POSCAR')
                os.chdir('../')
            os.chdir('../')

    # get the distortion information
    def get_distort_info(self, fin):
        """
        Loads distortion information from a specified file.
    
        This method reads data from a text file using NumPy's loadtxt function 
        and returns the contents as a NumPy array. The input file should contain 
        numerical data formatted appropriately for loading.
    
        Parameters:
        fin (str): The path to the input file containing distortion data.
    
        Returns:
        np.ndarray: An array containing the distortion information loaded from the file.
        """
        dsts = np.loadtxt(fin)
        return dsts
    
    # set the deformation matrix
    def set_deform_matrix(self, elist):
        """
        Sets the deformation tensor matrix based on the provided list of deformation values.
    
        Parameters:
        elist (list): A list containing six deformation values [e1, e2, e3, e4, e5, e6].
    
        Returns:
        np.matrix: A 3x3 deformation tensor matrix constructed from the input values.
        """
        #deformation tensor matrix
        e = elist
        e1 = e[0]
        e2 = e[1]
        e3 = e[2]
        e4 = e[3]
        e5 = e[4]
        e6 = e[5]
        d_matrix = np.matrix([
            [e1, 0.5 * e6, 0.5 * e5],
            [0.5 * e6, e2, 0.5 * e4],
            [0.5 * e5, 0.5 * e4, e3]
        ])
        return d_matrix
    
    # replace the original POSCAR file
    def replace_POSCAR_cell(self, cell_matrix, posold, posnew):
        """
        Replaces specific lines in a POSCAR file with new values from a given cell matrix.
    
        This method updates the 3rd, 4th, and 5th lines of the POSCAR file specified by `posold`
        with the corresponding values from the `cell_matrix`. The modified content is written to 
        a new file specified by `posnew`. 
    
        Parameters:
        cell_matrix (numpy.ndarray): A 3x3 matrix representing the new cell values.
        posold (str): The path to the original POSCAR file.
        posnew (str): The path where the modified POSCAR file will be saved.
    
        Raises:
        IOError: If there is an issue reading from `posold` or writing to `posnew`.
        ValueError: If `cell_matrix` does not have the correct size (must be 3x3).
        """
        #Replace the cell_matrix part of POSCAR by new one.
        if np.size(cell_matrix) != 9:
            sys.stderr.write('Error: Wrong input cell_matrix!\n')
        lines = open(posold).readlines()
        fw = open(posnew, 'w+')
        cnt = 1
        for line in lines:
            if cnt == 3 or cnt == 4 or cnt == 5:
                fw.write('%f\t%f\t%f\n' % (
                    cell_matrix[cnt - 3, 0], cell_matrix[cnt - 3, 1], cell_matrix[cnt - 3, 2]))
            else:
                fw.write(line)
            fw.flush()
            cnt += 1
        fw.close()
    
    # create the strained structure
    def strained_structure_create(self,args):
        """
        Creates strained structures based on the provided arguments.
    
        This method generates strained structures for a given number of states, 
        adjusting the number of strain steps and issuing warnings based on the 
        specified method (Energy or Stress) and maximum strain values. It handles 
        both general strained structure creation and specific NPT calculation 
        structures.
    
        Parameters:
        args (object): An object containing the following attributes:
                - nvolume: Optional; the volume number.
                - nstate: The number of states to process.
                - mode: The calculation mode ('NPT' or others).
                - nstrain: The number of strain steps.
                - maxstrain: The maximum strain value.
                - method: The method used for calculations ('Energy' or 'Stress').
                - lattice: The lattice structure to be used.
                - cij_order: The order of the elastic constants.
        Returns:
        None: The method generates strained structures for each state and writes
            status messages to standard output.
        """
        if args.nvolume == None:
            n = args.nstate
        else:
            n = args.nvolume

        sys.stdout.write(">>> DONE !\n")
        if args.mode != 'NPT':
            if args.nstrain <= 2 and args.method == 'Energy':
                args.nstrain = 9
                if args.maxstrain > 0.05:
                    sys.stdout.write('Warning: maxstrain = %f > 0.05 for Energy-Strain calculations\n' % args.maxstrain)
            elif args.nstrain <= 1 and args.method == 'Stress':
                args.nstrain = 8
                if args.maxstrain > 0.03:
                    sys.stdout.write('Warning: maxstrain = %f > 0.03 for Stress-Strain calculations\n' % args.maxstrain)
            if args.method == 'Stress' and args.nstrain % 2 == 1:
                args.nstrain = args.nstrain - 1
            sys.stdout.write('Start to create strained structures for each state.\n')
            sys.stdout.write('Using the maxstarin = %f and nstrain = %d for each kind of strain.\n' % (args.maxstrain, args.nstrain))
            strain_list = self.get_list(args.lattice, args.method, args.cij_order)
            sys.stdout.write('Strain Type:' + ', '.join(strain_list) + '\n')
            if args.nstrain % 2 == 1:
                et_range = np.linspace(-args.maxstrain, args.maxstrain, args.nstrain)
            else:
                et_range = np.linspace(-args.maxstrain, args.maxstrain, args.nstrain + 1)
                et_range = np.delete(et_range, int(args.nstrain / 2))
            for i in range(n):
                dirname = 'state_' + str(i+1)
                os.chdir(dirname)
                if os.path.exists('CONTCAR'):
                    shutil.copy2('CONTCAR', 'POSCAR')
                    sys.stdout.write('The POSCAR is updated by CONTCAR at %s\n' % dirname)
                self.apply(args.maxstrain, args.nstrain, args.lattice, args.method, args.cij_order)
                os.chdir('../')
            sys.stdout.write('Strained tructures for each sate have been prepared\n')
        else:
            sys.stdout.write('Start to create NPT calculation structures for each state.\n')
            for i in range(n):
                dirname = 'state_' + str(i+1)
                os.chdir(dirname)
                if os.path.exists('CONTCAR'):
                    shutil.copy2('CONTCAR', 'POSCAR')
                    sys.stdout.write('The POSCAR is updated by CONTCAR at %s\n' % dirname)
                if os.path.exists('NPT'):
                    shutil.rmtree('NPT')
                os.mkdir('NPT')
                os.chdir('NPT')
                shutil.copy2('../POSCAR', 'POSCAR')
                shutil.copy2('POSCAR', 'POSCAR_o')
                os.chdir('../../')
            sys.stdout.write('NPT calculation structures for each state have been prepared\n')

    # get the task number list
    def get_task_nlist(self, num):
        """
        Generates a list of string representations of numbers from 1 to the specified number.
    
        Parameters:
        num (int): The upper limit of the range (inclusive) for generating the list.
    
        Returns:
        list: A list of strings, where each string is a number from 1 to num.
        """
        DIR_LIST = []
        for i in range(1, num+1):
            DIR_LIST.append(str(i))
        return DIR_LIST

class Energy_Strain:
    """
    Energy_Strain class provides methods to analyze and fit energy-strain data for different crystal structures. 

    Attributes:
        __dir_es (str): Directory name for storing energy vs strain results.

    Methods:
        get_results_energy(LC='C', num_tasks='9'):
            Computes elastic constants using the energy-strain method and saves results to files.
        
        fit_res_energy(LC, method, cij_order, fitorder=6, plt=False):
            Fits the energy-strain data using polynomial fitting and optionally plots the results.
        
        plt_res_energy(Strain, Energy, fitres, figname, fmt):
            Plots the energy-strain data along with the fitted polynomial curve and saves the figure.
        
        format_Cij_energy(matrix_Aij, LC='C', fitorder=2, use_symmetry=False):
            Formats the Cij matrix from the energy-strain data based on symmetry considerations for various lattice types.
    """

    def __init__(self):
        self.__dir_es = 'Energy-vs-Strain'

    # get elastic constants from energy-strain method
    def get_results_energy(self,LC='C', num_tasks='9'):
        """
        Retrieves energy results for a specified material and number of tasks.
    
        This method iterates through a list of strain values, changing directories to 
        access energy data files, and extracts energy results from each directory. 
        The results are saved in a specified format for further analysis.
    
        Parameters:
        LC (str): The material type, default is 'C'.
        num_tasks (str): The number of tasks to process, default is '9'.
    
        Raises:
        SystemExit: If the expected directories or files do not exist.
        """
        method = 'Energy'
        cij_order = 2
        DstList = Strain().get_list(LC, method, cij_order)
 
        for dst in DstList:
            dirname = 'Dst_' + dst
            fdata = dirname + '_SE.dat'
            if os.path.exists(fdata):
                os.remove(fdata)
            if os.path.exists(dirname):
                os.chdir(dirname)
            else:
                sys.stderr.write('Error: No such directory %s at %s\n !' % (dirname, os.getcwd()))
                os.sys.exit(1)
 
            data = np.zeros((num_tasks, 2))
            strain = np.loadtxt(dirname + '.dat')
            for i in range(1,num_tasks+1):
                dir_path = str(i)
                if os.path.exists(dir_path):
                    os.chdir(dir_path)
                else:
                    sys.stderr.write('Error: No such directory %s at %s !\n' % (i, os.getcwd))
                    os.sys.exit(1)
                # get energy results in each directory.
                fin = 'OSZICAR'
                e = Common().extract_energy(fin)
                data[i-1, 0] = strain[i-1]
                data[i-1, 1] = e
                os.chdir('../')
            np.savetxt(fdata, data, fmt='%12.8f')
            os.chdir('../')

        # Directory management
        if (os.path.exists(self.__dir_es + '_old')):
            shutil.rmtree(self.__dir_es + '_old')
        if (os.path.exists(self.__dir_es)):
            os.rename(self.__dir_es, self.__dir_es + '_old')
        os.mkdir(self.__dir_es)
        os.chdir(self.__dir_es)
        for DST in glob.glob('../Dst_??/Dst_??_SE.dat'):
            shutil.copy2(DST,'.')
        os.chdir('../')
        return

    # fit the energy-strain data
    def fit_res_energy(self, LC, method, cij_order,  fitorder=6, plt=False):
        """
        Fit the residual energy as a function of strain using polynomial regression.
    
        Parameters:
        LC (list): List of loading conditions.
        method (str): Method to be used for strain calculation.
        cij_order (int): Order of the elastic constants.
        fitorder (int, optional): Order of the polynomial fit. Default is 6.
        plt (bool or str, optional): If set to "png" or "eps", plots the fit results. Default is False.
    
        Returns:
        numpy.ndarray: Coefficients of the polynomial fits for each strain-energy dataset.
        """
        os.chdir(self.__dir_es)
        
        # Initialize
        DstList = Strain().get_list(LC, method, cij_order)
        NoD = len(DstList)
        Coeff = np.zeros([NoD, fitorder + 1])
 
        for i in range(NoD):
            dst = DstList[i]
            dirtmp = 'Dst_' + dst
            fdata = dirtmp + '_SE.dat'
            data = np.loadtxt(fdata)
            strain = data[:, 0]
            energy = data[:, 1]
            Coeff[i, :] = np.polyfit(strain, energy, fitorder)
            if plt == "png":
                fitres = Coeff[i, :]
                figname = dirtmp + '.png'
                self.plt_res_energy(strain, energy, fitres, figname, "png")
            elif plt == "eps":
                fitres = Coeff[i, :]
                figname = dirtmp + '.eps'
                self.plt_res_energy(strain, energy, fitres, figname, "eps")
        os.chdir('../')
        return Coeff

    # plot the energy-strain data
    def plt_res_energy(self, Strain, Energy, fitres, figname, fmt):
        """
        Plots the relationship between strain and energy, fitting a polynomial curve to the data.
    
        Parameters:
        Strain (array-like): The strain values to be plotted.
        Energy (array-like): The corresponding energy values to be plotted.
        fitres (array-like): Coefficients of the polynomial fit.
        figname (str): The name of the file where the figure will be saved.
        fmt (str): The format in which to save the figure (e.g., 'png', 'pdf').

        Returns:
        None: The plot is generated and saved to the specified file.
        
        This function generates a plot of the strain versus energy, overlays a polynomial fit,
        and saves the figure to the specified file.
        """
        etmx = np.max(Strain) * 1.05
        polydata = np.poly1d(fitres)
        polystrain = np.linspace(-etmx, etmx, 100)
        polyenergy = polydata(polystrain)
        plt.plot(Strain, Energy, 'o', polystrain, polyenergy, '-')
        plt.grid(linestyle='--', linewidth=0.5)
        plt.xlabel('Strain')
        plt.ylabel('Energy Value (eV)')
        plt.savefig(figname, dpi=300, format=fmt)
        plt.close()

    # format the Cij from the energy-strain data using symmetry
    def format_Cij_energy(self, matrix_Aij, LC='C', fitorder=2, use_symmetry=False):
        """
        Formats the stiffness matrix Cij based on the provided matrix Aij and the specified lattice type.
    
        Parameters:
        matrix_Aij (numpy.ndarray): The input matrix containing coefficients for stiffness calculations.
        LC (str): The lattice type, which can be 'C' (Cubic), 'H' (Hexagonal), 'RI' (Rhombohedral I),
                  'RII' (Rhombohedral II), 'TI' (Tetragonal I), 'TII' (Tetragonal II),
                  'O' (Orthorhombic), 'M' (Monoclinic), or 'N' (Triclinic).
        fitorder (int): The order of fitting to be used for calculations (default is 2).
        use_symmetry (bool): If True, applies symmetry to the stiffness matrix (default is False).
    
        Returns:
        numpy.ndarray: The formatted stiffness matrix Cij of shape (6, 6).
        """
        # initialized Cij
        C = np.zeros([6, 6])
        #print(matrix_Aij)
        A2 = matrix_Aij[:, fitorder - 2]  # Need to approve.
 
        #print('>> A2 =\n')
        #print(A2)
 
        # Cubic structures
        if (LC == 'C' ):
            C[0, 0] = -2. * (A2[0] - 3. * A2[1]) / 3.
            C[1, 1] = C[0, 0]
            C[2, 2] = C[0, 0]
            C[3, 3] = A2[2] / 6.
            C[4, 4] = C[3, 3]
            C[5, 5] = C[3, 3]
            C[0, 1] = (2. * A2[0] - 3. * A2[1]) / 3.
            C[0, 2] = C[0, 1]
            C[1, 2] = C[0, 1]
 
        # Hexagonal structures
        if (LC == 'H' ):
            C[0, 0] = 2. * A2[3]
            C[0, 1] = 2. / 3. * A2[0] + 4. / 3. * A2[1] - 2. * A2[2] - 2. * A2[3]
            C[0, 2] = 1. / 6. * A2[0] - 2. / 3. * A2[1] + 0.5 * A2[2]
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[2, 2] = 2. * A2[2]
            C[3, 3] = -0.5 * A2[2] + 0.5 * A2[4]
            C[4, 4] = C[3, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Rhombohedral I structures
        if (LC == 'RI'):
            C[0, 0] = 2. * A2[3]
            C[0, 1] = A2[1] - 2. * A2[3]
            C[0, 2] = .5 * (A2[0] - A2[1] - A2[2])
            C[0, 3] = .5 * (-A2[3] - A2[4] + A2[5])
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[1, 3] = -C[0, 3]
            C[2, 2] = 2. * A2[2]
            C[3, 3] = .5 * A2[4]
            C[4, 4] = C[3, 3]
            C[4, 5] = C[0, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Rhombohedral II structures
        if (LC == 'RII'):
            C[0, 0] = 2. * A2[3]
            C[0, 1] = A2[1] - 2. * A2[3]
            C[0, 2] = .5 * (A2[0] - A2[1] - A2[2])
            C[0, 3] = .5 * (-A2[3] - A2[4] + A2[5])
            C[0, 4] = .5 * (-A2[3] - A2[4] + A2[6])
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[1, 3] = -C[0, 3]
            C[1, 4] = -C[0, 4]
            C[2, 2] = 2. * A2[2]
            C[3, 3] = .5 * A2[4]
            C[3, 5] = -C[0, 4]
            C[4, 4] = C[3, 3]
            C[4, 5] = C[0, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Tetragonal I structures
        if (LC == 'TI'):
            C[0, 0] = (A2[0] + 2. * A2[1]) / 3. + .5 * A2[2] - A2[3]
            C[0, 1] = (A2[0] + 2. * A2[1]) / 3. - .5 * A2[2] - A2[3]
            C[0, 2] = A2[0] / 6. - 2. * A2[1] / 3. + .5 * A2[3]
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[2, 2] = 2. * A2[3]
            C[3, 3] = .5 * A2[4]
            C[4, 4] = C[3, 3]
            C[5, 5] = .5 * A2[5]
 
        # Tetragonal II structures
        if (LC == 'TII'):
            C[0, 0] = (A2[0] + 2. * A2[1]) / 3. + .5 * A2[2] - A2[4]
            C[1, 1] = C[0, 0]
            C[0, 1] = (A2[0] + 2. * A2[1]) / 3. - .5 * A2[2] - A2[4]
            C[0, 2] = A2[0] / 6. - (2. / 3.) * A2[1] + .5 * A2[4]
            C[0, 5] = (-A2[2] + A2[3] - A2[6]) / 4.
            C[1, 2] = C[0, 2]
            C[1, 5] = -C[0, 5]
            C[2, 2] = 2. * A2[4]
            C[3, 3] = .5 * A2[5]
            C[4, 4] = C[3, 3]
            C[5, 5] = .5 * A2[6]
 
        # Orthorhombic structures
        if (LC == 'O'):
            C[0, 0] = 2. * A2[0] / 3. + 4. * A2[1] / \
                3. + A2[3] - 2. * A2[4] - 2. * A2[5]
            C[0, 1] = 1. * A2[0] / 3. + 2. * A2[1] / 3. - .5 * A2[3] - A2[5]
            C[0, 2] = 1. * A2[0] / 3. - 2. * A2[1] / \
                3. + 4. * A2[2] / 3. - .5 * A2[3] - A2[4]
            C[1, 1] = 2. * A2[4]
            C[1, 2] = -2. * A2[1] / 3. - 4. * A2[2] / \
                3. + .5 * A2[3] + A2[4] + A2[5]
            C[2, 2] = 2. * A2[5]
            C[3, 3] = .5 * A2[6]
            C[4, 4] = .5 * A2[7]
            C[5, 5] = .5 * A2[8]
 
        # Monoclinic structures
        if (LC == 'M'):
            C[0, 0] = 2. * A2[0] / 3. + 8. * \
                (A2[1] + A2[2]) / 3. - 2. * (A2[5] + A2[8] + A2[9])
            C[0, 1] = A2[0] / 3. + 4. * (A2[1] + A2[2]) / 3. - 2. * A2[5] - A2[9]
            C[0, 2] = (A2[0] - 4. * A2[2]) / 3. + A2[5] - A2[8]
            C[0, 5] = -1. * A2[0] / 6. - 2. * \
                (A2[1] + A2[2]) / 3. + .5 * \
                (A2[5] + A2[7] + A2[8] + A2[9] - A2[12])
            C[1, 1] = 2. * A2[8]
            C[1, 2] = -4. * (2. * A2[1] + A2[2]) / 3. + 2. * \
                A2[5] + A2[8] + A2[9] 
            C[1, 5] = -1. * A2[0] / 6. - 2. * \
                (A2[1] + A2[2]) / 3. - .5 * A2[4] + \
                A2[5] + .5 * (A2[7] + A2[8] + A2[9])
            C[2, 2] = 2. * A2[9]
            C[2, 5] = A2[3]  - A2[9] - A2[12]
            C[3, 3] = .5 * A2[10]
            C[3, 4] = .25 * (A2[6] - A2[10] - A2[11])
            C[4, 4] = .5 * A2[11]
            C[5, 5] = .5 * A2[12]
 
        # Triclinic structures
        if (LC == 'N'):
            C[0, 0] = 2. * A2[0]
            C[0, 1] = 1. * (-A2[0] - A2[1] + A2[6])
            C[0, 2] = 1. * (-A2[0] - A2[2] + A2[7])
            C[0, 3] = .5 * (-A2[0] - A2[3] + A2[8])
            C[0, 4] = .5 * (-A2[0] + A2[9] - A2[4])
            C[0, 5] = .5 * (-A2[0] + A2[10] - A2[5])
            C[1, 1] = 2. * A2[1]
            C[1, 2] = 1. * (A2[11] - A2[1] - A2[2])
            C[1, 3] = .5 * (A2[12] - A2[1] - A2[3])
            C[1, 4] = .5 * (A2[13] - A2[1] - A2[4])
            C[1, 5] = .5 * (A2[14] - A2[1] - A2[5])
            C[2, 2] = 2. * A2[2]
            C[2, 3] = .5 * (A2[15] - A2[2] - A2[3])
            C[2, 4] = .5 * (A2[16] - A2[2] - A2[4])
            C[2, 5] = .5 * (A2[17] - A2[2] - A2[5])
            C[3, 3] = .5 * A2[3]
            C[3, 4] = .25 * (A2[18] - A2[3] - A2[4])
            C[3, 5] = .25 * (A2[19] - A2[3] - A2[5])
            C[4, 4] = .5 * A2[4]
            C[4, 5] = .25 * (A2[20] - A2[4] - A2[5])
            C[5, 5] = .5 * A2[5]
        if use_symmetry:
            for i in range(5):
                for j in range(i + 1, 6):
                    C[j, i] = C[i, j]
        else :
            pass
        return C

class Stress_Strain:
    """
    Stress_Strain class provides methods to analyze and manipulate stress-strain data.
    It includes functionalities to extract elastic constants, fit stress-strain relations,
    plot results, and format stiffness matrices based on symmetry for various crystal structures.

    Methods:
    - get_results_stress(LC, mode, MLFF, cij_order=2): Extracts stress-strain results from specified directories,
    writes data to files, and manages directory structure.
    
    - fit_res_stress(LC, cij_order=2, fitorder=1, plt=False): Fits stress-strain data using polynomial regression
    and saves the coefficients and residuals to files.

    - plt_res_stress(Strain, Stress, fitres, figname): Plots stress-strain data along with the fitted curve
    and saves the figure to the specified filename.

    - format_Cij_stress(matrix_Aij, LC='C', use_symmetry=False): Formats the stiffness matrix based on the
    specified lattice class (LC) and applies symmetry if required.
    """
    
    def __init__(self):
        self.__dir_ss = 'Stress-vs-Strain'

    # get elastic constants from stress-strain method
    def get_results_stress(self,LC, mode, MLFF, cij_order=2):
        """
        Computes the stress results based on Lagrangian strain for a given loading condition (LC) and mode.
    
        Parameters:
        LC (str): The loading condition identifier.
        mode (str): The mode of stress calculation ('cold', 'QSA', or 'NVT').
        MLFF (str): The machine learning force field to be used.
        cij_order (int, optional): The order of the Cauchy stress tensor, default is 2.
    
        This method retrieves the Lagrangian strain list and corresponding distortion information,
        navigates through directories for each strain state, and calculates the physical stresses
        based on the provided mode. The results are written to output files in Voigt notation.
    
        Raises:
        SystemExit: If the specified directory does not exist or if the deformation is too large.
        """
        mthd = 'Stress'
        Lag_strain_list = Strain().get_list(LC, mthd, cij_order)
        distort_dic = Strain().list()
        for sidx in Lag_strain_list:
            Ls_list = distort_dic[sidx]
            Dstn = 'Dst_' + sidx
            if (os.path.exists(Dstn)):
                os.chdir(Dstn)
            else:
                sys.stderr.write('Error: No such directory %s at %s !\n' % (Dstn, os.getcwd()))
                os.sys.exit(1)
    
            # Writing the Strain-Stress information
            flstres = open(Dstn + '_LS.dat', 'w')
            fpstres = open(Dstn + '_PS.dat', 'w')
    
            flstres.write('# Lagrangian strain and Lagrangian stresses (LS) in Voigt notation for ' +
                          Dstn + '.\n')
            flstres.write(
                '# Lag. strain         LS1          LS2          LS3          LS4          LS5          LS6\n')
    
            fpstres.write('# Lagrangian strain and physical stresses (PS) in Voigt notation for ' +
                          Dstn + '.\n')
            fpstres.write(
                '# Lag. strain         PS1          PS2          PS3          PS4          PS5          PS6\n')
    
            # NoP, number of etas
            distort_strain = Strain().get_distort_info(Dstn + '.dat')
            for j in range(1, len(distort_strain) + 1):
                Dstn_num = str(j)
    
                if (os.path.exists(Dstn_num)):
                    os.chdir(Dstn_num)
    
                    strain_j = distort_strain[j-1]
    
                    le = np.zeros(6)
                    for i in range(6):
                        le[i] = Ls_list[i]
                    Lv = strain_j * le
    
                    #--- Lag. to phy. strain (eta = eps + 0.5*eps*esp)
                    eta_matrix = np.zeros((3, 3))
                    eta_matrix[0, 0] = Lv[0]
                    eta_matrix[0, 1] = Lv[5] / 2.
                    eta_matrix[0, 2] = Lv[4] / 2.
    
                    eta_matrix[1, 0] = Lv[5] / 2.
                    eta_matrix[1, 1] = Lv[1]
                    eta_matrix[1, 2] = Lv[3] / 2.
    
                    eta_matrix[2, 0] = Lv[4] / 2.
                    eta_matrix[2, 1] = Lv[3] / 2.
                    eta_matrix[2, 2] = Lv[2]
    
                    norm = 1.0
                    eps_matrix = eta_matrix
                    if (np.linalg.norm(eta_matrix) > 0.7):
                        sys.stderr('Error: Too large deformation!\n')
                    sig = np.zeros((3, 3))
                    if os.path.exists('OUTCAR'):
                        Common().extract_stress('OUTCAR', mode, MLFF)
                        if mode == 'cold':
                            sig = Common().read_stress(fin='stress.out', mode='cold')
                        elif mode == 'QSA':
                            sig = Common().read_stress(fin='stress.out', mode='QSA')
                        elif mode == 'NVT':
                            sig = Common().read_stress(fin='stress.out', mode='NVT')
    
                    while(norm > 1.e-10):
                        x = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.
                        norm = np.linalg.norm(x - eps_matrix)
                        eps_matrix = x
    
                    #--- Calculating the deformation matrix -----------------------
                    i_matrix = np.array([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])
                    def_matrix = i_matrix + eps_matrix
    
                    #--- Reading the physical stresses from the output files ------
                    idm = np.linalg.inv(def_matrix)
                    tao = np.linalg.det(def_matrix) * np.dot(idm, np.dot(sig, idm))
                    if (strain_j > 0):
                        strain = '+%12.10f\t'
                    else:
                        strain = '%13.10f\t'
                    sigstr = strain
                    sigstr += '%14.8f\t' * 6
                    sigstr += '\n'
                    fpstres.write(sigstr % (strain_j, sig[0, 0], sig[1, 1], sig[
                                  2, 2], sig[1, 2], sig[0, 2], sig[0, 1]))
                    taostr = strain
                    taostr += '%14.8f\t' * 6
                    taostr += '\n'
                    flstres.write(sigstr % (strain_j, tao[0, 0], tao[1, 1], tao[
                                  2, 2], tao[1, 2], tao[0, 2], tao[0, 1]))
                    os.chdir('../')
                else:
                    sys.stderr.write('Error: No such directory %s at %s !\n' % (Dstn_num, os.getcwd))
                    os.sys.exit(1)

            flstres.close()
            fpstres.close()
            os.chdir('../')
    
        # Directory management
        if (os.path.exists(self.__dir_ss + '_old')):
            shutil.rmtree(self.__dir_ss + '_old')
        if (os.path.exists(self.__dir_ss)):
            os.rename(self.__dir_ss, self.__dir_ss + '_old')
        os.mkdir(self.__dir_ss)
        os.chdir(self.__dir_ss)
        for DST in glob.glob('../Dst_??/Dst_??_LS.dat'):
            shutil.copy2(DST,'.')
        for DST in glob.glob('../Dst_??/Dst_??_PS.dat'):
            shutil.copy2(DST,'.')
        os.chdir('../')

    # fit the stress-strain data !notcheckfinished
    def fit_res_stress(self,LC, cij_order=2, fitorder=1, plt=False):
        """
        Fits the stress-strain data using polynomial regression and saves the coefficients and residuals to files.

        Parameters:
        LC (str): The loading condition identifier.
        cij_order (int, optional): The order of the Cauchy stress tensor, default is 2.
        fitorder (int, optional): The order of the polynomial fit, default is 1.
        plt (bool or str, optional): If set to "png" or "eps", plots the fit results. Default is False.

        Returns:
        numpy.ndarray: Coefficients of the polynomial fits for each stress-strain dataset.
        """
        Lag_strain_list = Strain().get_list(LC, 'Stress', cij_order)
        os.chdir(self.__dir_ss)
        # Get A1 for each stress-strain relations
        Cs = np.zeros((6, 6))
        Cres = np.zeros((6, 6))
        i = 0
        for sidx in Lag_strain_list:
            Dstn = 'Dst_' + sidx
            # loading the data
            fdata = Dstn + '_PS.dat'
            data = np.loadtxt(fdata)
            strain = data[:, 0]
            ndata = len(strain)
            if ndata < fitorder:
                sys.exit("number of strains (%d) < fitorder (%d)!" %
                         (ndata, fitorder))
            # collect coeffs of first deriviative
            for j in range(6):
                cname = 'Eq' + str(i + 1) + str(j + 1)
                stress = data[:, j + 1]
                results = np.polyfit(strain, stress, fitorder, full=True)
                '''
                if fitorder= n and coeff1= p
                p[0]*x[i]**n+p[1]*x[i]**(n-1)+...+p[n-1]*x[i] + p[n] = yi
                p[n] is the shift value.
                '''
                coeff1 = results[0]
                A1 = coeff1[fitorder - 1]
                res = Common().calc_CV(strain, stress, coeff1)
                Cs[i, j] = A1
                Cres[i, j] = res
                if plt == "png":
                    figname = cname + '.png'
                    self.plt_res_stress(strain, stress, coeff1, figname, "png")
                elif plt == "eps":
                    figname = cname + '.eps'
                    self.plt_res_stress(strain, stress, coeff1, figname, "eps")
            i += 1
 
        # Write information
        fcs = open('Cs.dat', 'w')
        fcres = open('Cres.dat', 'w')
        for i in range(6):
            for j in range(6):
                fcs.write('%8.4f\t' % Cs[i, j])
                fcres.write('%8.4f\t' % Cres[i, j])
            fcs.write('\n')
            fcres.write('\n')
        fcs.close()
        fcres.close()
        os.chdir('../')
        return Cs

    # plot the stress-strain data
    def plt_res_stress(self, Strain, Stress, fitres, figname):
        """
        Plots the stress-strain curve and saves the figure as a PNG file.
    
        Parameters:
        Strain (array-like): The strain values to be plotted.
        Stress (array-like): The corresponding stress values to be plotted.
        fitres (array-like): Coefficients for the polynomial fit of the stress-strain data.
        figname (str): The filename for saving the plot, including the file extension.

        Returns:
        None: The function saves the plot to the specified file.

        The function generates a plot of the provided strain and stress data, overlays a polynomial fit,
        and saves the resulting figure to the specified filename.
        """
        etmx = np.max(Strain) * 1.05
        polydata = np.poly1d(fitres)
        polystrain = np.linspace(-etmx, etmx, 100)
        polyenergy = polydata(polystrain)
        plt.plot(Strain, Stress, 'o', polystrain, polyenergy, '-')
        plt.grid(linestyle='--', linewidth=0.3)
        plt.xlabel('Strain')
        plt.ylabel('Stress Value (kbar)')
        plt.legend([figname.split('.')[0]])
        plt.savefig(figname, dpi=300, format='png')
        plt.close()
        return

    # format the Cij from the stress-strain data using symmetry
    def format_Cij_stress(self, matrix_Aij, LC='C', use_symmetry= False):
        """
        Formats the stiffness matrix C based on the input matrix Aij and the specified lattice type (LC).
        
        Parameters:
        matrix_Aij (numpy.ndarray): Input matrix containing stiffness coefficients.
        LC (str): Lattice type, which can be 'C' (Cubic), 'H' (Hexagonal), 'RI' (Rhombohedral I),
                  'RII' (Rhombohedral II), 'TI' (Tetragonal I), 'TII' (Tetragonal II),
                  'O' (Orthorhombic), 'M' (Monoclinic), or 'N' (Triclinic).
        use_symmetry (bool): If True, applies symmetry to the stiffness matrix.
    
        Returns:
        numpy.ndarray: The formatted stiffness matrix C.
        """
        A = matrix_Aij
        C=np.zeros([6,6])
        if (LC == 'C' ):
            C[0, 0] = A[0, 0]
            C[1, 1] = C[0, 0]
            C[2, 2] = C[0, 0]
            C[3, 3] = A[0, 5]
            C[4, 4] = C[3, 3]
            C[5, 5] = C[3, 3]
            C[0, 1] = .5 * (A[0, 1] + A[0, 2])
            C[0, 2] = C[0, 1]
            C[1, 2] = C[0, 1]
 
        # Hexagonal structures
        if (LC == 'H' ):
            C[0, 0] = A[1, 0]
            C[0, 1] = A[1, 1]
            C[0, 2] = .5 * (A[0, 0] + A[1,2])
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[2, 2] = A[0, 2]
            C[3, 3] = A[1, 3]
            C[4, 4] = C[3, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Rhombohedral I structures
        if (LC == 'RI'):
            C[0, 0] = A[1, 1] - A[0, 4]
            C[0, 1] = A[1, 0] + A[0, 4]
            C[0, 2] = .5 * (A[0, 0] + A[0, 1])
            C[0, 3] = A[0, 4]
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[1, 3] =-C[0, 3]
            C[2, 2] = A[0, 2]
            C[3, 3] = A[1, 3] + A[0, 4]
            C[4, 4] = C[3, 3]
            C[4, 5] = C[0, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Rhombohedral II structures
        if (LC == 'RII'):
            C[0, 0] = A[1, 1] - A[0, 4]
            C[0, 1] = A[1, 0] + A[0, 4]
            C[0, 2] = .5 * (A[0, 0] + A[0, 1])
            C[0, 3] = A[0, 4]
            C[0, 4] =-A[0, 3]
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[1, 3] =-C[0, 3]
            C[1, 4] =-C[0, 4]
            C[2, 2] = A[0, 2]
            C[3, 3] = A[1, 3] + A[0, 4]
            C[3, 5] =-C[0, 4]
            C[4, 4] = C[3, 3]
            C[4, 5] = C[0, 3]
            C[5, 5] = .5 * (C[0, 0] - C[0, 1])
 
        # Tetragonal I structures
        if (LC == 'TI'):
            C[0, 0] = 2 * A[0, 1] - A[1, 2]
            C[0, 1] = 2 * A[0, 0] - A[1, 2]
            C[0, 2] = .5 * A[1, 2]
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[2, 2] = A[0, 2] - .5 * C[0, 2]
            C[3, 3] = A[0, 3]
            C[4, 4] = C[3, 3]
            C[5, 5] = A[1, 5]
 
        # Tetragonal II structures
        if (LC == 'TII'):
            C[0, 0] = 2 * A[0, 1] - A[1, 2]
            C[0, 1] = 2 * A[0, 0] - A[1, 2]
            C[0, 2] = .5 * A[1, 2]
            C[0, 5] = .5 * (A[1, 0] - A[1, 1])
            C[1, 1] = C[0, 0]
            C[1, 2] = C[0, 2]
            C[1, 5] =-C[0, 5]
            C[2, 2] = A[0, 2] - .5 * C[0, 2]
            C[3, 3] = A[0, 3]
            C[4, 4] = C[3, 3]
            C[5, 5] = A[1, 5]
 
        # Orthorhombic structures
        """
        if (LC == 'O'):
            C[0, 0] = A[2, 0] - A[1, 0]
            C[0, 1] = A[2, 0] + A[1, 0]
            C[0, 2] = .5 * (A[2, 2] - A[1, 2] + A[0, 0])
            C[1, 1] = A[1, 1] + A[2, 1]
            C[1, 2] = .5 * (A[2, 2] + A[1, 2] + A[0, 1]) 
            C[2, 2] = A[0, 2] 
            C[3, 3] = A[2, 3] 
            C[4, 4] = A[1, 4] 
            C[5, 5] = A[0, 5]
        """
        if (LC == 'O'):
            C[0, 0] = A[0,0]
            C[0, 1] = .5 * (A[0,1] + A[1,0])
            C[0, 2] = .5 * (A[0,2] + A[2,0])
            C[1, 1] = A[1,1]
            C[1, 2] = .5 * (A[1,2] + A[2,1])
            C[2, 2] = A[2,2]
            C[3, 3] = A[0,3]
            C[4, 4] = A[1,4]
            C[5, 5] = A[2,5]
        # Monoclinic structures
        if (LC == 'M'):
            C[0, 0] = A[2, 0]
            C[0, 1] = .5 * (A[2, 1] + A[3, 0])
            C[0, 2] = .5 * (A[2, 2] + A[0, 0])
            C[0, 5] = A[1, 0] 
            C[1, 1] = A[3, 1]
            C[1, 2] = .5 * (A[0, 1] + A[3, 2])
            C[1, 5] = A[1, 1]
            C[2, 2] = A[0, 2]
            C[2, 5] = .5 * (A[1, 2] + A[1, 2])
            C[3, 3] = A[2, 3]
            C[3, 4] = .5 * (A[2, 4] + A[3, 3])
            C[4, 4] = A[3, 4]
            C[5, 5] = A[1, 5]
 
        # Triclinic structures
        if (LC == 'N'):
            C[0, 0] = A[0, 0]
            C[0, 1] = A[0, 1]
            C[0, 2] = A[0, 2]
            C[0, 3] = A[0, 3]
            C[0, 4] = A[0, 4]
            C[0, 5] = A[0, 5]
            C[1, 1] = A[1, 1]
            C[1, 2] = A[1, 2]
            C[1, 3] = A[1, 3]
            C[1, 4] = A[1, 4]
            C[1, 5] = A[1, 5]
            C[2, 2] = A[2, 2]
            C[2, 3] = A[2, 3]
            C[2, 4] = A[2, 4]
            C[2, 5] = A[2, 5]
            C[3, 3] = A[3, 3]
            C[3, 4] = A[3, 4]
            C[3, 5] = A[3, 5]
            C[4, 4] = A[4, 4]
            C[4, 5] = A[4, 5]
            C[5, 5] = A[5, 5]
        if use_symmetry:
            for i in range(5):
                for j in range(i + 1, 6):
                    C[j, i] = C[i, j]
        else :
            pass
        return C

class NPT:
    """
    NPT class implements methods for performing NPT (constant Number of particles, Pressure, and Temperature) simulations.

    Methods:
    - adapative_sampling_algorithm(limit, volume_file, pressure_file):
        Calculates the correlation between simulation steps, volume, and pressure. It adjusts the starting point based on the volume ratio compared to the average volume.

    - extract_structure():
        Extracts structural information from the XDATCAR file and writes it to structure.out.

    - extract_strain(start, end):
        Computes the strain from the extracted structure data and saves it to strain.out.

    - solve_cij(lattice, start, end):
        Calculates the elastic constants (Cij) based on the strain and stress data. The method handles different lattice types and computes the corresponding stiffness matrix.
    """
    
    # calculate correlation between step with volume, pressure and temperature
    def adapative_sampling_algorithm(self, limit = 1.03, volume_file='volume.out', pressure_file='pressure.out'):
        """
        Performs adaptive sampling based on volume and pressure data.
    
        This method reads volume and pressure data from specified files, calculates the average volume,
        and determines the starting point for adaptive sampling based on a specified limit. It iteratively
        adjusts the starting index until the ratio of volumes to the average volume falls below the limit.
    
        Parameters:
        limit (float): The threshold ratio for adaptive sampling (default is 1.03).
        volume_file (str): The path to the file containing volume data (default is 'volume.out').
        pressure_file (str): The path to the file containing pressure data (default is 'pressure.out').
    
        Returns:
        tuple: A tuple containing the starting index and the ending index of the data used for sampling.
    
        Raises:
        SystemExit: If the specified volume or pressure files do not exist.
        """
        if os.path.exists(volume_file) and os.path.exists(pressure_file):
            volumes = np.loadtxt(volume_file)
            pressure = np.loadtxt(pressure_file)
            steps = np.arange(len(volumes))
            #data = np.vstack((steps, volumes, pressure)).T
            data = np.vstack((steps, volumes)).T
            v_average = np.average(data[:,1])
            ratio = volumes/v_average
            start = 0
            end = len(data)-1
            #correlation_matrix = np.corrcoef(data, rowvar=False)
            #print('Correlation between step and volume: ', correlation_matrix[0, 1])
            #print('Correlation between step and pressure: ', correlation_matrix[0, 2])
            while ratio[start] > limit:
                #print('Start: ', start, ratio[start])
                for i in range(len(ratio)):
                    if ratio[i] > limit:
                        start = i
                    else:
                        start = i
                        break
                v_average = np.average(data[start:,1])
                ratio = volumes/v_average
            else:
                pass
            #print('Start: ', start, ratio[start])
            #correlation_matrix = np.corrcoef(data[start:end, :], rowvar=False)
            #print('Correlation between step and volume: ', correlation_matrix[0, 1])
            #print('Correlation between step and pressure: ', correlation_matrix[0, 2])


            #left = start
            #right = int(len(data) * 0.5)-1
            #mid = int(left + (right - left) / 2) - 1
            #correlation_matrix_left = np.corrcoef(data[left:], rowvar=False)
            #correlation_matrix_right = np.corrcoef(data[right:], rowvar=False)
            #correlation_matrix_mid = np.corrcoef(data[mid:], rowvar=False)
            #if correlation_matrix_left[0, 1] > -0.1 and correlation_matrix_left[0, 2] < 0.1:
            #    pass
            #else:

            #    n_start = int(len(data) * 0.5)-1
            #    def find_optimal_n_start(data):
            #        left, right = 0, n_start
            #        while left < right:
            #            mid = (left + right + 1) // 2
            #            correlation_matrix_segment = np.corrcoef(data[mid:], rowvar=False)
            #            step_vs_volume = correlation_matrix_segment[0, 1]
            #            step_vs_pressure = correlation_matrix_segment[0, 2]
            #            if step_vs_volume > -0.1 and step_vs_pressure < 0.1:
            #                left = mid
            #            else:
            #                right = mid - 1
            #        return left

            #    optimal_n_start = find_optimal_n_start(data)


            #    data_new = data[optimal_n_start:]
            #    correlation_matrix = np.corrcoef(data_new, rowvar=False) 
            #    print('Optimal n_start: ', optimal_n_start)
            #    print('Correlation between step and volume: ', correlation_matrix[0, 1])
            #    print('Correlation between step and pressure: ', correlation_matrix[0, 2])
            #    print('Correlation between step and temperature: ', correlation_matrix[0, 3])
        else:
            sys.stderr.write('Error: volume.out, pressure.out or temperature.out is not exist at ' + os.getcwd() + '. Please finish calculation first!\n')
            os.sys.exit(1)
        return start, end

    # extract the structure information from XDATCAR
    def extract_structure(self):
        """
        Extracts the last three direct configuration coordinates from the XDATCAR file 
        and writes them to the structure.out file. The function checks for the existence 
        of the XDATCAR file before attempting to read from it. If the file does not exist, 
        an error message is printed to stderr and the program exits.
    
        The function utilizes a deque to maintain a sliding window of the last five lines 
        read from the input file, allowing it to access the necessary lines when the 
        'Direct configuration' line is encountered.

        Returns:
        None: The function writes the extracted structure information to the structure.out file.

        Raises:
        SystemExit: If the XDATCAR file does not exist.
        """
        if os.path.exists("XDATCAR"):
            with open('XDATCAR', 'r') as inputfile, open('structure.out', 'w') as outputfile:
                queue = deque(maxlen=5)
                for line in inputfile:
                    if 'Direct configuration' in line:
                        temp1 = queue[0].split()
                        temp2 = queue[1].split()
                        temp3 = queue[2].split()

                        outputfile.write( temp1[0] + ' ' + temp1[1] + ' ' + temp1[2] + ' '
                                        + temp2[0] + ' ' + temp2[1] + ' ' + temp2[2] + ' '
                                        + temp3[0] + ' ' + temp3[1] + ' ' + temp3[2] + '\n')
                    queue.append(line)
        elif os.path.exists("structure.out"):
            pass
        else:
            sys.stderr.write('Error: XDATCAR file is not exist at ' + os.getcwd() +'. Please finish calculation first!\n')
            os.sys.exit(1)
    
    # extract the strain information from structure.out
    def extract_strain(self, start, end):
        """
        Extracts the strain tensor from a series of lattice structures.
    
        Parameters:
        start (int): The starting index for the lattice data.
        end (int): The ending index for the lattice data.

        Returns:
        None: The function writes the calculated strain tensor to the strain.out file.
        
        This method reads lattice data from 'structure.out', computes the average lattice 
        between the specified start and end indices, and calculates the strain tensor 
        for each lattice configuration in the specified range. The resulting strain 
        values are saved to 'strain.out'.
    
        The strain tensor is represented in a 6-component format, where:
        - strain[:,0]: ε_xx
        - strain[:,1]: ε_yy
        - strain[:,2]: ε_zz
        - strain[:,3]: ε_xy + ε_yx
        - strain[:,4]: ε_xz + ε_zx
        - strain[:,5]: ε_yz + ε_zy
        """
        num = end - start
        strain = np.zeros((num, 6))
        lattice = np.loadtxt('structure.out')
        lattice = lattice.reshape((len(lattice), 3, 3))
        lattice_average = lattice[start:end].mean(axis=0)
        lattice_inverse = np.linalg.inv(lattice_average)
        strain_matrix = np.zeros((num,3,3))
        for i in range(num):
            strain_matrix[i] = np.dot(lattice_inverse, lattice[i+start]) - np.identity(3)
        strain_matrix = strain_matrix.reshape((num, 9))
        strain[:,0] = strain_matrix[:,0]
        strain[:,1] = strain_matrix[:,4]
        strain[:,2] = strain_matrix[:,8]
        strain[:,3] = strain_matrix[:,5] + strain_matrix[:,7]
        strain[:,4] = strain_matrix[:,2] + strain_matrix[:,6]
        strain[:,5] = strain_matrix[:,1] + strain_matrix[:,3]
        np.savetxt('strain.out', strain)

    # calculate the elastic constants from NPT method
    def solve_cij(self, lattice, start, end):
        """
        Calculate the elastic constants (Cij) for different lattice types based on strain and stress data.
    
        Parameters:
        lattice (str): The type of lattice ('C', 'H', 'RI', 'RII', 'TI', 'TII', 'O', 'M', 'N').
        start (int): The starting index for the stress and strain data.
        end (int): The ending index for the stress and strain data.
    
        Returns:
        numpy.ndarray: A 6x6 matrix containing the calculated elastic constants.
        
        The function reads strain and stress data from 'strain.out' and 'stress.out' files, respectively.
        It processes the data based on the specified lattice type and computes the elastic constants using
        least squares fitting. The function handles multiple lattice types with specific formulations for each.
        """

        strain = np.loadtxt('strain.out')
        stress = np.loadtxt('stress.out')
        stress = -stress[start:end,:]

        mask = np.all(np.abs(strain) < 0.02, axis=1)
        #strain = strain[mask]
        #stress = stress[mask]

        for i in range(6):
            stress[:,i] = stress[:,i] - np.average(stress[:,i])
        cij = np.zeros((6, 6))

        index = len(stress[:,0])
        S_stress = np.zeros((6*index,1))
        for i in np.arange(0,index):
            S_stress[6*i+0,0]=stress[i,0]
            S_stress[6*i+1,0]=stress[i,1]
            S_stress[6*i+2,0]=stress[i,2]
            S_stress[6*i+3,0]=stress[i,4]
            S_stress[6*i+4,0]=stress[i,5]
            S_stress[6*i+5,0]=stress[i,3]
            i = i+1
        if lattice == 'C':
            S_strain = np.zeros((6*index,3))
            for i in np.arange(0,index):
                S_strain[6*i+0,0]=strain[i,0]
                S_strain[6*i+0,1]=strain[i,1]+strain[i,2]
                S_strain[6*i+1,0]=strain[i,1]
                S_strain[6*i+1,1]=strain[i,0]+strain[i,2]
                S_strain[6*i+2,0]=strain[i,2]
                S_strain[6*i+2,1]=strain[i,0]+strain[i,1]
                S_strain[6*i+3,2]=strain[i,3]
                S_strain[6*i+4,2]=strain[i,4]
                S_strain[6*i+5,2]=strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
                     
            cij[0,0] = solve_elas[0]
            cij[0,1] = solve_elas[1]
            cij[0,2] = cij[0,1]
            cij[0,3] = 0.0
            cij[0,4] = 0.0 
            cij[0,5] = 0.0
            cij[1,0] = cij[0,1] 
            cij[1,1] = cij[0,0]
            cij[1,2] = cij[0,1]
            cij[1,3] = 0.0
            cij[1,4] = 0.0
            cij[1,5] = 0.0
            cij[2,0] = cij[0,1]
            cij[2,1] = cij[0,1]
            cij[2,2] = cij[0,0]
            cij[2,3] = 0.0
            cij[2,4] = 0.0
            cij[2,5] = 0.0
            cij[3,0] = 0.0
            cij[3,1] = 0.0
            cij[3,2] = 0.0
            cij[3,3] = solve_elas[2]
            cij[3,4] = 0.0
            cij[3,5] = 0.0
            cij[4,0] = 0.0
            cij[4,1] = 0.0
            cij[4,2] = 0.0
            cij[4,3] = 0.0
            cij[4,4] = cij[3,3]
            cij[4,5] = 0.0
            cij[5,0] = 0.0
            cij[5,1] = 0.0
            cij[5,2] = 0.0
            cij[5,3] = 0.0
            cij[5,4] = 0.0
            cij[5,5] = cij[3,3]
        elif lattice == 'H':
            S_strain = np.zeros((6*index,6))
            for i in np.arange(0,index):
                S_strain[6*i+0,0]=strain[i,0]; S_strain[6*i+0,1]=strain[i,1]
                S_strain[6*i+0,2]=strain[i,2]
                S_strain[6*i+1,0]=strain[i,1]; S_strain[6*i+1,1]=strain[i,0]
                S_strain[6*i+1,2]=strain[i,2]
                S_strain[6*i+2,2]=strain[i,0]+strain[i,1]; S_strain[6*i+2,3]=strain[i,2]
                S_strain[6*i+3,3]=strain[i,3]
                S_strain[6*i+4,4]=strain[i,4]
                S_strain[6*i+5,5]=strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            cij[0,0] = solve_elas[0]
            cij[0,1] = solve_elas[1]
            cij[0,2] = solve_elas[2]
            cij[0,3] = 0.0
            cij[0,4] = 0.0
            cij[0,5] = 0.0
            cij[1,0] = cij[0,1]
            cij[1,1] = cij[0,0]
            cij[1,2] = cij[0,2]
            cij[1,3] = 0.0
            cij[1,4] = 0.0
            cij[1,5] = 0.0
            cij[2,0] = cij[0,2]
            cij[2,1] = cij[1,2]
            cij[2,2] = solve_elas[3]
            cij[2,3] = 0.0
            cij[2,4] = 0.0
            cij[2,5] = 0.0
            cij[3,0] = 0.0
            cij[3,1] = 0.0
            cij[3,2] = 0.0
            cij[3,3] = solve_elas[4]
            cij[3,4] = 0.0
            cij[3,5] = 0.0
            cij[4,0] = 0.0
            cij[4,1] = 0.0
            cij[4,2] = 0.0
            cij[4,3] = 0.0
            cij[4,4] = cij[3,3]
            cij[4,5] = 0.0
            cij[5,0] = 0.0
            cij[5,1] = 0.0
            cij[5,2] = 0.0
            cij[5,3] = 0.0
            cij[5,4] = 0.0
            cij[5,5] = solve_elas[5]
        elif lattice == 'RI' or lattice == 'RII':
            S_strain = np.zeros((6*index,7))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3]=strain[i,0:3]; S_strain[6*i+0,6]=strain[i,5]
                S_strain[6*i+1,1]=strain[i,0]; S_strain[6*i+1,0]=strain[i,1]
                S_strain[6*i+1,2]=strain[i,2]; S_strain[6*i+1,6]=-strain[i,5]
                S_strain[6*i+2,2]=strain[i,0]+strain[i,1]; S_strain[6*i+2,3]=strain[i,2]
                S_strain[6*i+3,4]=strain[i,3]; S_strain[6*i+4,4]=strain[i,4]
                S_strain[6*i+5,5]=strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            cij[0,0] = solve_elas[0]
            cij[0,1] = solve_elas[1]
            cij[0,2] = solve_elas[2]
            cij[0,3] = 0.0
            cij[0,4] = 0.0
            cij[0,5] = solve_elas[6]
            cij[1,0] = cij[0,1]
            cij[1,1] = cij[0,0]
            cij[1,2] = cij[0,2]
            cij[1,3] = 0.0
            cij[1,4] = 0.0
            cij[1,5] = -cij[0,5]
            cij[2,0] = cij[0,2]
            cij[2,1] = cij[1,2]
            cij[2,2] = solve_elas[3]
            cij[2,3] = 0.0
            cij[2,4] = 0.0
            cij[2,5] = 0.0
            cij[3,0] = 0.0
            cij[3,1] = 0.0
            cij[3,2] = 0.0
            cij[3,3] = solve_elas[4]
            cij[3,4] = 0.0
            cij[3,5] = 0.0
            cij[4,0] = 0.0
            cij[4,1] = 0.0
            cij[4,2] = 0.0
            cij[4,3] = 0.0
            cij[4,4] = cij[3,3]
            cij[4,5] = 0.0
            cij[5,0] = cij[0,5]
            cij[5,1] = -cij[0,5]
            cij[5,2] = 0.0
            cij[5,3] = 0.0
            cij[5,4] = 0.0
            cij[5,5] = solve_elas[5]
        elif lattice == 'TI' or lattice == 'TII':
            S_strain = np.zeros((6*index,7))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:5]=strain[i,0:5]
                S_strain[6*i+1,0]=strain[i,1]; S_strain[6*i+1,1]=strain[i,0]
                S_strain[6*i+1,2]=strain[i,2]; S_strain[6*i+1,3]=-strain[i,3]
                S_strain[6*i+1,4]=-strain[i,4]
                S_strain[6*i+2,2]=strain[i,0]+strain[i,1]
                S_strain[6*i+2,5]=strain[i,2]
                S_strain[6*i+3,3]=strain[i,0]-strain[i,1]
                S_strain[6*i+3,4]=-strain[i,5]; S_strain[6*i+3,6]=strain[i,3]
                S_strain[6*i+4,3]=strain[i,5]; S_strain[6*i+4,6]=strain[i,4]
                S_strain[6*i+4,4]=strain[i,0]-strain[i,1]
                S_strain[6*i+5,0]=strain[i,5]/2; S_strain[6*i+5,1]=-strain[i,5]/2
                S_strain[6*i+5,3]=strain[i,4]; S_strain[6*i+5,4]=-strain[i,5]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            cij[0,0] = solve_elas[0]
            cij[0,1] = solve_elas[1]
            cij[0,2] = solve_elas[2]
            cij[0,3] = solve_elas[3]
            cij[0,4] = solve_elas[4]
            cij[0,5] = 0.0
            cij[1,0] = cij[0,1]
            cij[1,1] = cij[0,0]
            cij[1,2] = cij[0,2]
            cij[1,3] = -cij[0,3]
            cij[1,4] = -cij[0,4]
            cij[1,5] = 0.0
            cij[2,0] = cij[0,2]
            cij[2,1] = cij[1,2]
            cij[2,2] = solve_elas[5]
            cij[2,3] = 0.0
            cij[2,4] = 0.0
            cij[2,5] = 0.0
            cij[3,0] = cij[0,3]
            cij[3,1] = -cij[0,3]
            cij[3,2] = 0.0
            cij[3,3] = solve_elas[6]
            cij[3,4] = 0.0
            cij[3,5] = -cij[0,4]
            cij[4,0] = cij[0,4]
            cij[4,1] = -cij[0,4]
            cij[4,2] = 0.0
            cij[4,3] = 0.0
            cij[4,4] = solve_elas[7]
            cij[4,5] = cij[0,3]
            cij[5,0] = 0.0
            cij[5,1] = 0.0
            cij[5,2] = 0.0
            cij[5,3] = -cij[0,4]
            cij[5,4] = cij[0,3]
            cij[5,5] = (cij[0,0]-cij[0,1])/2
        elif lattice == 'O':
            S_strain = np.zeros((6*index,9))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:3]=strain[i,0:3]
                S_strain[6*i+1,1]=strain[i,0]; S_strain[6*i+1,3:5]=strain[i,1:3]
                S_strain[6*i+2,2]=strain[i,0]; S_strain[6*i+2,4:6]=strain[i,1:3]
                S_strain[6*i+3,6]=strain[i,3]; S_strain[6*i+4,7]=strain[i,4]
                S_strain[6*i+5,8]=strain[i,5]
        elif lattice == 'M':
            S_strain = np.zeros((6*index,13))
            for i in np.arange(0,index):
                S_strain[6*i+0,0]=strain[i,0]; S_strain[6*i+0,1]=strain[i,1]
                S_strain[6*i+0,2]=strain[i,2]; S_strain[6*i+0,3]=strain[i,4]
                S_strain[6*i+1,1]=strain[i,0]; S_strain[6*i+1,4]=strain[i,1]
                S_strain[6*i+1,5]=strain[i,2]; S_strain[6*i+1,6]=strain[i,4]
                S_strain[6*i+2,2]=strain[i,0]; S_strain[6*i+2,5]=strain[i,1]
                S_strain[6*i+2,7]=strain[i,2]; S_strain[6*i+2,8]=strain[i,4]
                S_strain[6*i+3,9]=strain[i,3]; S_strain[6*i+3,10]=strain[i,5]
                S_strain[6*i+4,3]=strain[i,0]; S_strain[6*i+4,6]=strain[i,1]
                S_strain[6*i+4,8]=strain[i,2]; S_strain[6*i+4,11]=strain[i,4]
                S_strain[6*i+5,10]=strain[i,3]; S_strain[6*i+5,12]=strain[i,5]
        elif lattice == 'N':
            S_strain = np.zeros((6*index,21))
            for i in np.arange(0,index):
                S_strain[6*i+0,0:6]=strain[i,0:6]
                S_strain[6*i+1,1] = strain[i,0]; S_strain[6*i+1,6:11] = strain[i,1:6]#to be confirmed
                S_strain[6*i+2,2] = strain[i,0]; S_strain[6*i+2,7]=strain[i,1]
                S_strain[6*i+2,11:15]=strain[i,2:6]
                S_strain[6*i+3,3] = strain[i,0]; S_strain[6*i+3,8]=strain[i,1]
                S_strain[6*i+3,12]=strain[i,2]; S_strain[6*i+3,15:18]=strain[i,3:6]
                S_strain[6*i+4,4]=strain[i,0]; S_strain[6*i+4,9]=strain[i,1]
                S_strain[6*i+4,13]=strain[i,2]; S_strain[6*i+4,16]=strain[i,3]
                S_strain[6*i+4,18:20] = strain[i,4:6]
                S_strain[6*i+5,5] = strain[i,0]; S_strain[6*i+5,10] = strain[i,1]
                S_strain[6*i+5,14] = strain[i,2]; S_strain[6*i+5,17] = strain[i,3]
                S_strain[6*i+5,19:21]=strain[i,4:6]
            solve_elas = np.linalg.lstsq(S_strain, S_stress,rcond=-1)[0]
            cij[0,0] = solve_elas[0]
            cij[0,1] = solve_elas[1]
            cij[0,2] = solve_elas[2]
            cij[0,3] = solve_elas[3]
            cij[0,4] = solve_elas[4]
            cij[0,5] = solve_elas[5]
            cij[1,0] = solve_elas[1]
            cij[1,1] = solve_elas[6]
            cij[1,2] = solve_elas[7]
            cij[1,3] = solve_elas[8]
            cij[1,4] = solve_elas[9]
            cij[1,5] = solve_elas[10]
            cij[2,0] = solve_elas[2]
            cij[2,1] = solve_elas[7]
            cij[2,2] = solve_elas[11]
            cij[2,3] = solve_elas[12]
            cij[2,4] = solve_elas[13]
            cij[2,5] = solve_elas[14]
            cij[3,0] = solve_elas[3]
            cij[3,1] = solve_elas[8]
            cij[3,2] = solve_elas[12]
            cij[3,3] = solve_elas[15]
            cij[3,4] = solve_elas[16]
            cij[3,5] = solve_elas[17]
            cij[4,0] = solve_elas[4]
            cij[4,1] = solve_elas[9]
            cij[4,2] = solve_elas[13]
            cij[4,3] = solve_elas[16]
            cij[4,4] = solve_elas[18]
            cij[4,5] = solve_elas[19]
            cij[5,0] = solve_elas[5]
            cij[5,1] = solve_elas[10]
            cij[5,2] = solve_elas[14]
            cij[5,3] = solve_elas[17]
            cij[5,4] = solve_elas[19]
            cij[5,5] = solve_elas[20]
        else:
            pass
        return cij
