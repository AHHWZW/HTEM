import argparse
import os
import sys
import ast

class Basic_para:
    """
    Basic_para class encapsulates fundamental physical constants and valid run styles for simulations.

    Attributes:
        run_style_list (list): A list of valid calculation modes including 'create', 'strain', 'relax2', 
                            'static', 'NPT', 'NVT', 'get_results', 'anisotropy', 'gen', 'model', 
                            and 'update'.
        Na (float): Avogadro's number, representing the number of particles in one mole.
        e (float): Elementary charge, the charge of a single proton.
        Ang (float): Conversion factor from Angstroms to meters.
        Bohr (float): Bohr radius in meters, a physical constant related to atomic structure.
        Ryd2eV (float): Conversion factor from Rydberg units to electronvolts.
        Ryd2GPa (float): Conversion factor from Rydberg per cubic atomic unit to gigapascals.
        ev2GPa (float): Conversion factor from electronvolts per cubic Angstrom to gigapascals.
        GPa (float): Conversion factor from gigapascals to pascals.
        hbar (float): Reduced Planck's constant, a fundamental constant in quantum mechanics.
        kB (float): Boltzmann constant, relating temperature to energy.
        DMAX (int): Maximum dimension constant used in calculations.
    """

    def __init__(self):
        self.run_style_list = ['create',
                        'strain',
                        'relax2',
                        'static',
                        'NPT',
                        'NVT',
                        #'clean',
                        'get_results',
                        'anisotropy',
                        'gen',
                        'model',
                        'update']
        self.Na = 6.02214129e23              # Avogadro constant
        self.e = 1.602176565e-19             # elementary charge
        self.Ang = 1e-10                     # Ang to meter
        self.Bohr = 5.291772086e-11          # Bohr to meter
        self.Ryd2eV = 13.605698066           # Ryd to eV
        self.Ryd2GPa = (self.e * self.Ryd2eV) / (1e9 * self.Bohr**3)  # Ryd/[a.u.^3] to GPa
        self.ev2GPa = self.e / pow(self.Ang, 3) / 1e9
        self.GPa = 1e9                       # GPa to Pa
        self.hbar = 1.054571726e-34          # reduced Planck constant
        self.kB = 1.3806488e-23              # Boltzmann constant
        self.DMAX = 62

class Input_para:
    """
    Input_para class handles the input parameters for the HTEM (High-throughput Toolkits for Elasticity Modeling) application. It provides methods for formatting parameters, parsing command line arguments, and reading/writing configuration files.

    Attributes:
        runstyle (str): The style of the run.
        code (str): The code used for calculations, default is 'VASP'.
        nstrain (int): Number of strained structures.
        ksetting (float or list): KSPACING or KMESH settings.
        encut (int): Energy cutoff for calculations.
        sigma (float): Smearing width for calculations.
        lattice (str): Type of lattice.
        sgn (int): Space Group Number.
        method (str): Calculation method, e.g., 'Energy' or 'Stress'.
        cij_order (int): Order of cij for fitting.
        forder (int): Fitting order of solution.
        maxstrain (float): Maximum strain value.
        inputfile (str): Name of the input file.
        nvolume (int): Number of calculated volumes.
        natom (int): Number of atoms.
        M (float): Averaged atomic mass.
        plt (str): Plotting mode.
        version (str): Version of the application.
        build (list): Build parameters.
        supercell (list): Supercell dimensions.
        mode (str): Calculation mode.
        MLFF (bool): Flag for MLFF mode.
        modulus (str): Modulus type for anisotropy calculations.
        temperature (float or list): Temperature settings.
        pressure (float or list): Pressure settings.
        T_ref (float): Reference temperature for SAM.
        upmode (str): Update mode for input files.
        T_range (list): Temperature range for modeling.
        P_range (list): Pressure range for modeling.
        weight (float): Weight coefficient for SAM.
        read_file (str): File to read elasticity data from.

    Methods:
        format_para: Converts input parameters to their corresponding types.
        command_input: Parses command line arguments and assigns them to attributes.
        read_config: Reads parameters from a configuration file and assigns them to attributes.
        write_config: Writes parameters to a configuration file.
    """
    
    def __init__(self):
        self.runstyle = None
        self.code = 'VASP'
        self.nstrain = None
        self.ksetting = None
        self.encut = None
        self.sigma = None
        self.lattice = None
        self.sgn = None
        self.method = None
        self.cij_order = 2
        self.forder = None
        self.maxstrain = None
        self.inputfile = None
        self.nvolume = None
        self.nstate = None
        self.natom = None
        self.M = None
        self.plt = None
        self.version = None
        self.build = None
        self.supercell = None
        self.mode = None
        self.MLFF = None
        self.modulus = None
        #self.amode = None 
        self.temperature = None
        self.pressure = None
        self.T_ref = None
        self.upmode = None
        self.T_range = None
        self.P_range = None
        self.weight = None
        self.read_file = None

        self.list_args = {'c':'code',
                    'ns':'nstrain',
                    'k':'ksetting',
                    'e':'encut',
                    's':'sigma',
                    'lt':'lattice',
                    'sgn':'sgn',
                    'mthd':'method',
                    'corder':'cij_order',
                    'forder':'forder',
                    'ms':'maxstrain',
                    'in':'inputfile',
                    'nv':'nvolume',
                    'na':'natom',
                    'nstate':'nstate',
                    'M':'M',
                    'b':'build',
                    'mode':'mode',
                    'T':'temperature',
                    'P':'pressure',
                    'plt':'plt',
                    'sc':'supercell',
                    'MLFF':'MLFF',
                    'modulus':'modulus',
                    #'amode':'amode',
                    'T0':'T_ref',
                    'up':'upmode',
                    'Tr':'T_range',
                    'Pr':'P_range',
                    'weight':'weight',
                    'read':'read_file'}
        
        self.int_list = ['nstrain',
                         'encut',
                         'sgn',
                         'cij_order',
                         'forder',
                         'nvolume',
                         'nstate',
                         'natom']
        self.float_list = ['maxstrain',
                           'sigma',
                           'M',
                           'T_ref',
                            'weight']
        self.str_list = ['code',
                            'lattice',
                            'method',
                            'inputfile',
                            'version',
                            'mode',
                            'plt',
                            'modulus',
                            #'amode',
                            'upmode',
                            'plt',
                            'read_file']
        self.mode_list = ['cold',
                            'QSA',
                            'NVT',
                            'NPT']
        self.bool_list = ['MLFF']

    # format the parameters
    def format_para(self):
        """
        Formats input parameters by converting them to their corresponding types.
        
        This method processes several lists of attributes (int_list, float_list, str_list, 
        bool_list) and converts their values to the appropriate types. It also handles 
        specific cases for 'supercell', 'build', 'temperature', 'pressure', and 'ksetting' 
        attributes, ensuring they meet the expected formats and constraints. Errors are 
        reported to standard error if the parameters do not conform to the required 
        specifications.
    
        Raises:
        ValueError: If the parameters do not meet the expected format or type.
        """
        for arg in self.int_list:
            if getattr(self, arg) != None:
                setattr(self, arg, int(getattr(self, arg)))
            else:
                pass
        for arg in self.float_list:
            if getattr(self, arg) != None:
                setattr(self, arg, float(getattr(self, arg)))
            else:
                pass
        for arg in self.str_list:
            if getattr(self, arg) != None:
                setattr(self, arg, str(getattr(self, arg)))
            else:
                pass
        for arg in self.bool_list:
            if getattr(self, arg) != None:
                if isinstance(getattr(self, arg), str):
                    if getattr(self, arg).lower() == 'true':
                        setattr(self, arg, True)
                    elif getattr(self, arg).lower() == 'false':
                        setattr(self, arg, False)
                    else:
                        sys.stderr.write('Error: the parameters of -%s should be True or False!\n' % arg)
            else:
                pass
        for arg in ['supercell']:
            if getattr(self, arg) != None:
                if isinstance(getattr(self, arg), list) and len(getattr(self, arg)) == 3:
                    setattr(self, arg, [int(i) for i in getattr(self, arg)])
                else:
                    sys.stderr.write('Error: the parameters of -sc(--supercell) should be three values separated by commas!\n')
            else:
                pass
        for arg in ['build']:
            if getattr(self, arg) != None:
                if isinstance(getattr(self, arg), list) and len(getattr(self, arg)) == 5:
                    setattr(self, arg, [str(getattr(self, arg)[0]), str(getattr(self, arg)[1]), float(getattr(self, arg)[2]), float(getattr(self, arg)[3]), float(getattr(self, arg)[4])])
                else:
                    sys.stderr.write('Error: the parameters of -b(--build) should be five values separated by commas!\n')
            else:
                pass
        for arg in ['temperature','pressure']:
            if getattr(self, arg) != None:
                if isinstance(getattr(self, arg), list) and len(getattr(self, arg)) == 3:
                    setattr(self, arg, [float(getattr(self, arg)[0]), float(getattr(self, arg)[1]), int(getattr(self, arg)[2])])
                elif not isinstance(getattr(self, arg), list):
                    setattr(self, arg, float(getattr(self, arg)))
                else:
                    sys.stderr.write('Error: the parameters of -T(--temperature) and -P(--pressure) should be one or three values separated by commas!\n')
        for arg in ['ksetting']:
            if getattr(self, arg) != None:
                if isinstance(getattr(self, arg), list) and len(getattr(self, arg)) == 3:
                    setattr(self, arg, [int(i) for i in getattr(self, arg)])
                elif not isinstance(getattr(self, arg), list):
                    setattr(self, arg, float(getattr(self, arg)))
                else:
                    sys.stderr.write('Error: the parameters of -k(--ksetting) should be one or three values separated by commas!\n')

    # command line input
    def command_input(self):
        """
        Command input parser for the HTEM application.
        
        This method handles the parsing of command line arguments for the HTEM (High-throughput Toolkits for Elasticity Modeling) application. It defines several helper functions to convert input values for temperature, pressure, and ksetting into appropriate types. The method sets up argument groups for basic parameters, input parameters, calculation methods, plot parameters, anisotropy parameters, model parameters, and update state files parameters.
        
        Each argument is defined with its expected type, default value, and help description. The parsed arguments are then assigned to the corresponding attributes of the class instance.

        Raises:
        argparse.ArgumentTypeError: If the input values do not meet the expected format for temperature, pressure, or ksetting.
        
        Notes:
            - The method uses the argparse module to parse command line arguments.
            - It defines type conversion functions for temperature, pressure, and ksetting.
            - The parsed arguments are assigned to the corresponding attributes of the class instance.
        """

        # Define type conversion functions for temperature, pressure, and ksetting
        def parse_type_temperature(value):
            values = value.split(',')
            if len(values) == 1:
                return float(values[0])
            elif len(values) == 3:
                return [float(values[0]), float(values[1]), int(values[2])]
            elif values == None:
                pass
            else:
                raise argparse.ArgumentTypeError("Input for temperature must be one or three values separated by commas!\n")
        
        def parse_type_pressure(value):
            values = value.split(',')
            if len(values) == 1:
                return float(values[0])
            elif len(values) == 3:
                return [float(values[0]), float(values[1]), int(values[2])]
            elif values == None:
                pass
            else:
                raise argparse.ArgumentTypeError("Input for pressure must be one or three values separated by commas!\n")
        
        def parse_type_ksetting(value): 
            values = value.split(',')
            if len(values) == 1:
                return float(value)
            elif len(values) == 3:
                return [int(i) for i in values]
            elif values == None:
                pass
            else:
                raise argparse.ArgumentTypeError("Input for ksetting must be one or three values separated by commas!\n")
        
        """
        def parse_type_supercell(value):
            values = value.split(',')
            if len(values) == 3:
                return [int(i) for i in values]
            elif values == None:
                pass
            else:
                raise argparse.ArgumentTypeError("Input for supercell must be three values separated by commas!\n")
        
        def parse_type_build(value):
            values = value.split(',')
            if len(values) == 5:
                return [str(values[0]), str(values[1]), float(values[2]), float(values[3]), float(values[4])]
            elif values == None:
                pass 
            else:
                raise argparse.ArgumentTypeError("Input for build must be five values separated by commas!\n")
        """

        # parse the command line arguments
        para = argparse.ArgumentParser(
            description= "HTEM: High-throughput Toolkits for Elasticity Modeling")#"HTEM: High-Throughput Elasticity and Thermodynamics of Materials")
        basic_group = para.add_argument_group('Basic Parameters')
        basic_group.add_argument('runstyle',
                            help='one of arguments in '
                            + str(Basic_para().run_style_list)
                            + ' should be input',
                            choices = Basic_para().run_style_list,
                            metavar='runstyle')
        basic_group.add_argument('-mode', '--mode', type=str, default='cold',
                            help='Choose the calculation mode: cold, QSA, NVT, NPT',
                            choices=['cold', 'QSA', 'NVT', 'NPT'],
                            metavar='cold/QSA/NVT/NPT')
        
        input_group = para.add_argument_group('Parameters for automatically generate input files for HTEM')
        input_group.add_argument('-k', '--ksetting', type=parse_type_ksetting, default=0.1,
                            help="KSPACING or KMESH, eg. 0.1 for KSPACING or 4,4,4 for KMESH",
                            metavar='0.1/4,4,4 ')
        input_group.add_argument('-T', '--temperature', type=parse_type_temperature, default=0,
                            help='Temperature of the system, unit K, eg. 300 for 300K or 300,500,3 for 300-500K with 3 points and 100K step size',
                            metavar='300/300,500,3 ')
        input_group.add_argument('-P', '--pressure', type=parse_type_pressure, default=0,
                            help='Pressure of the system, unit GPa, eg. 0 for 0GPa or 0,10,3 for 0-10GPa with 3 points and 5GPa step size',
                            metavar='0/0,10,3 ')
        input_group.add_argument('-lt', '--lattice', type=str,
                            help="Choose the lattice type: C, H, TI, TII, RI, RII, O, M, N, can be read from POSCAR",
                            choices=['C', 'H', 'TI', 'TII', 'RI', 'RII', 'O', 'M', 'N'],
                            metavar='C/H/TI/TII/RI/RII/O/M/N')
        input_group.add_argument('-in', '--inputfile', type=str,
                            help='Name of input file',
                            metavar='state.in')
        input_group.add_argument('-e', '--encut', type=int, default=500,
                            help="ENCUT of INCAR file",
                            metavar='500')
        input_group.add_argument('-s', '--sigma', type=float, default=0.20,
                            help="SIGMA of INCAR file, 0.05~0.30",
                            metavar='0.20')
        input_group.add_argument('-M', '--M', type=float,
                            help='Averaged atomic mass, can be read from POSCAR')
        input_group.add_argument('-na','--natom', type=int,
                            help='The number of atoms of POSCAR, can be read from POSCAR')
        input_group.add_argument('-sgn', type=int,
                            help='Space Group Number in 1~229, can be read from POSCAR')

        input_group.add_argument('-MLFF', '--MLFF', type=bool, default=False,
                           help='Choose the MLFF mode for INCAR file: True, False',
                           choices=['True', 'False'],
                           metavar='True/False')

        
        method_group = para.add_argument_group('Parameters related to the calculation method')
        method_group.add_argument('-mthd', '--method', type=str, default='Energy',
                            help='Choose the calculation method: Energy, Stress. Energy is only supported by mode cold and QSA',
                            choices=['Energy', 'Stress'],
                            metavar='Energy/Stress')
        method_group.add_argument('-ns', '--nstrain', type=int, default=9,
                            help="Number of strained structures for each strain.",
                            metavar='9') 
        method_group.add_argument('-forder', type=int,
                            help='Fitting order of solution: default 3 for Strain-Energy and 2 for Strain-Stress',
                            metavar='3/2')
        method_group.add_argument('-ms', '--maxstrain', type=float, default=0.02,
                            help='Max strain, eg. 0.01~0.05',
                            metavar='0.02')

        method_group.add_argument('-nstate','--nstate', type=int,
                            help='The number of calculated states, can be read from inputfile')

        plt_group = para.add_argument_group('Plot Parameters')
        plt_group.add_argument('-plt', type=str,
                            help='Choose the plot mode: Faslse, png, eps',
                            choices=['False', 'png', 'eps'],
                            metavar='False/png/eps')
        plt_group.add_argument('-read', '--read_file', type=str,
                            help='Read the elasticity data from the file',
                            metavar='file')

        anisotropy_group = para.add_argument_group('Anisotropy Parameters')
        anisotropy_group.add_argument('-modulus', '--modulus', type=str,
                            help='Choose the modulus to calculate the anisotropy: B, G, E, nu, sound. This parameter is for mode plot',
                            choices=['B', 'G', 'E', 'nu', 'sound', 'gif'],
                            metavar='B/G/E/nu/sound/gif')

        
        model_group = para.add_argument_group('Model Parameters')
        model_group.add_argument('-T0', '--T_ref', type=float,
                            help='The reference temperature for SAM',
                            metavar='0')
        model_group.add_argument('-Tr', '--T_range', type=parse_type_temperature,
                            help='Temperature range for SAM elasticity modeling, eg. 0,2000,101 for 0-2000K with 101 points and 20K step size',
                            metavar='0,2000,101 ')
        model_group.add_argument('-Pr', '--P_range', type=parse_type_pressure,
                            help='Pressure range for SAM elasticity modeling, eg. 0,10,101 for 0-10GPa with 101 points and 0.1GPa step size',
                            metavar='0,10,101 ')
        model_group.add_argument('-weight', type=float, default=2,
                            help='The weight coefficient of SAM, please set it between 1 and 2',
                            metavar='2')
        
        
        update_group = para.add_argument_group('Update state files parameters')
        update_group.add_argument('-up', '--upmode', type=str,
                            help='Update the inputfile\'s volume and density for each state, eg. for state.in\', two mode: CONTCAR, NPT',
                            choices=['CONTCAR', 'NPT'],
                            metavar='CONTCAR/NPT')
        """
        unused_group = para.add_argument_group('Unused Parameters')
        unused_group.add_argument('-sc','--supercell', type = parse_type_supercell,
                            help="Build the supercell, eg. 2,2,2, not supported yet",
                            metavar='2,2,2 ')
        unused_group.add_argument('-b','--build', type = parse_type_build,
                            help="Build the basic structure, eg. fcc,Al4,3.5,3.5,3.5, not supported yet",
                            metavar='fcc,Al4,3.5,3.5,3.5 ')
        unused_group.add_argument('-nv','--nvolume', type=int,
                            help='The number of calculated volumes, not supported yet',
                            metavar='9') 
        unused_group.add_argument('-c', '--code', type=str, default='vasp',
                            help='default code is VASP, other codes are not supported yet',
                            metavar='VASP') 
        unused_group.add_argument('-corder', '--cij_order', type=int, default=2,
                            help='Set the order of cij, default is 2, other is not supported yet',
                            metavar='2')
        """
        args = para.parse_args() 
        
        self.runstyle = args.runstyle
        
        # Assigning values to variables based on command line input
        s_d_args = [i.strip("-") for i in sys.argv[1:] if i.startswith('-') and not i.startswith('--')]
        d_d_args = [i.strip("--") for i in sys.argv[1:] if i.startswith('--')]
        for key in self.list_args.keys():
            if key in s_d_args:
                setattr(self, self.list_args[key], getattr(args, self.list_args[key]))
            elif key in d_d_args:
                setattr(self, key, getattr(args, key))
            elif self.list_args[key] not in ['nvolume','supercell','build','code','cij_order'] and getattr(args, self.list_args[key]) == para.get_default(self.list_args[key]) and getattr(self, self.list_args[key]) == None:
                setattr(self, self.list_args[key], getattr(args, self.list_args[key]))
            else:
                pass

    # read the config file
    def read_config(self):
        """
        Reads configuration parameters from a 'config' file and sets them as attributes of the instance.
    
        The configuration file should contain lines in the format 'parameter=value'. 
        If the value is a list (indicated by square brackets), it will be evaluated to a Python list.
        Only parameters that are present in the instance's list_args will be set.
    
        If the 'config' file does not exist, no action is taken.
    
        Attributes set:
        self.list_args (dict): A dictionary containing valid parameter names.
    
        Raises:
        FileNotFoundError: If the 'config' file does not exist (handled internally).
        """
        if os.path.exists('config'):
            fn = open('config','r').read()
            lines = fn.split('\n')
            for line in lines:
                if line == '': continue
                para = line.split('=')[0].strip()
                value = line.split('=')[1].strip()
                if para in self.list_args.values():
                    if value.startswith('['):
                        value = ast.literal_eval(value)
                    setattr(self, para, value)
                else:
                    pass
        else:
            pass
    
    # write the config file
    def write_config(self, *args):
        """
        Writes configuration parameters to a 'config' file.
    
        This method updates existing parameters or adds new ones based on the provided arguments.
        It reads the current configuration from the 'config' file, modifies it according to the 
        specified parameters, and writes the changes back to the file. If the 'config' file does 
        not exist, it creates a new one.
    
        Parameters:
        *args: Variable length argument list of parameter names to be written to the config file.
    
        Raises:
        FileNotFoundError: If the 'config' file does not exist, it is created.
    
        Notes:
            - The method checks if each parameter exists in the instance's list of arguments.
            - It handles both updating existing parameters and adding new ones.
            - Errors are reported to stderr for any undefined parameters.
        """
        if os.path.exists('config'):
            fn = open('config','r+')
            lines = fn.readlines()
            paras = []
            fn.seek(0)
            fn.truncate()
            for i,line in enumerate(lines):
                para = line.split('=')[0].strip()
                if para not in args and para in self.list_args.values():
                    lines[i] = line
                    paras.append(para)
                    fn.write(lines[i])
                elif para in args and para in self.list_args.values():
                    lines[i] = f'{para} = {getattr(self, para)}\n'
                    paras.append(para)
                    fn.write(lines[i])
                    #sys.stdout.write('Config is update %s = %s \n' % (para, getattr(self, para)))
                else:
                    sys.stderr.write('Error: the parameter %s is not defined! Skip it! \n' % para)
                    pass
            for arg in args:
                if arg not in paras:
                    if arg in self.list_args.values():
                        line = f'{arg} = {getattr(self, arg)}\n'
                        fn.write(line)
                        #sys.stdout.write('Config is update %s = %s \n' % (para, getattr(self, arg)))
                    else:
                        sys.stderr.write('Error: the parameter %s is not defined! Skip it! \n' % arg)
                        pass
            fn.close()
        else:
            fn = open('config','a+')
            for arg in args:
                if arg in args:
                    line = f'{arg} = {getattr(self, arg)}\n'
                    fn.write(line)
                else:
                    sys.stderr.write('Error: the parameter %s is not defined! Skip it! \n' % arg)
                    pass
            fn.close()
