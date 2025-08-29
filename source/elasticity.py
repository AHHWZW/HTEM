import numpy as np
import sys
import os

from .parameter import Basic_para
from .write_output import write_output

class Elasticity:
    """
    Elasticity class for calculating and managing the elastic properties of materials.

    This class provides methods to initialize material properties, compute stiffness and compliance matrices,
    and derive various elastic moduli based on the material's characteristics. It supports reading data from
    external files and formatting elasticity matrices based on material symmetry.

    Attributes:
        V (float): Volume of the material.
        M (float): Mass of the material.
        rho (float): Density of the material.
        P (float): Pressure of the material.
        T (float): Temperature of the material.
        system (str): Type of the material system.
        C_matrix (np.array): Elastic stiffness matrix.
        S_matrix (np.array): Elastic compliance matrix.
        BV, GV, EV (float): Bulk, shear, and Young's moduli.
        BR, GR, ER (float): Bulk, shear, and Young's moduli from the compliance matrix.
        BH, GH, EH (float): Hardness-related moduli.
        C_matrix_Fedorov (np.array): Fedorov transformation of the stiffness matrix.
        S_matrix_Fedorov (np.array): Compliance matrix in Fedorov form.
        nuH (float): Poisson's ratio related to hardness.
        kH (float): Shear modulus-to-bulk modulus ratio.
        AVR, AU (float): Volume and Universal anisotropy factors.
        HH (float): Hardness from Chen et al.'s formula.
        CL, CB (float): Longitudinal and bulk sound velocities.
        Debye (float): Debye temperature.

    Methods:
        C(i, j): Returns the value of the stiffness matrix for given indices.
        S(i, j): Returns the value of the compliance matrix for given indices.
        init_Fedorov_matrix(): Initializes the Fedorov matrix and its inverse based on the stiffness matrix.
        read_output(fn, args, i): Reads material elasticity data from a file and updates class attributes.
        format_Cij(symmetry, E_input): Formats the elasticity matrix based on the material's symmetry.
        cal_properties(args): Calculates the elastic properties based on the stiffness matrix.
    """

    def __init__(self):
        self.V = None
        self.M = None
        self.rho = None
        self.P = None
        self.T = None
        self.system = None
        self.C_matrix = None
        self.S_matrix = None
        self.BV = None
        self.GV = None
        self.EV = None
        self.BR = None
        self.GR = None
        self.ER = None
        self.BH = None
        self.GH = None
        self.EH = None
        self.C_matrix_Fedorov = None
        self.S_matrix_Fedorov = None
        self.nuH = None
        self.kH = None
        self.AVR = None
        self.AU = None
        self.HH = None
        self.CL = None
        self.CB = None
        self.Debye = None

    # elastic stiffness matrixs
    def C(self, i, j):
        """
        Retrieves the specified element from the stiffness matrix.
    
        This method accesses the stiffness matrix (C_matrix) using the provided
        indices (i, j) and returns the corresponding value. The indices are 
        expected to be 1-based, and the method adjusts them to 0-based indexing 
        for internal access.
    
        Parameters:
        i (int): The row index of the stiffness matrix (1-based).
        j (int): The column index of the stiffness matrix (1-based).
    
        Returns:
        float: The value at the specified position in the stiffness matrix.
        """
        return self.C_matrix[i-1, j-1]
    
    # elastic compliance matrix
    def S(self, i, j):
        """
        Retrieves the value from the S_matrix at the specified indices.
    
        Parameters:
        i (int): The row index (1-based).
        j (int): The column index (1-based).
    
        Returns:
        The value at the specified position in the S_matrix.
        """
        return self.S_matrix[i-1, j-1]
    
    # Fedorov matrix
    def init_Fedorov_matrix(self):
        """
        Initializes the Fedorov matrix based on the current C_matrix.
    
        This method checks if the C_matrix is entirely zero. If it is not, it constructs the 
        C_matrix_Fedorov by populating it according to specific rules based on the indices 
        of the matrix. The resulting Fedorov matrix is a 6x6 matrix where:
        - The upper left 3x3 block is filled with values from the C function.
        - The lower right 3x3 block is filled with double the values from the C function.
        - The off-diagonal blocks are filled with the square root of 2 times the values from the C function.
    
        Finally, it computes the inverse of the C_matrix_Fedorov and stores it in S_matrix_Fedorov.
        
        Attributes set:
        C_matrix_Fedorov (np.array): Fedorov transformation of the stiffness matrix.
        S_matrix_Fedorov (np.array): Compliance matrix in Fedorov form.
        """
        if np.all(self.C_matrix[:,:] == 0):
            pass
        else:
            self.C_matrix_Fedorov = np.zeros((6,6))
            for i in range(0, 6):
                for j in range(0, 6):
                    if i <= 2 and j <= 2:
                        self.C_matrix_Fedorov[i, j] = self.C(i+1, j+1)
                    elif i >= 3 and j >= 3:
                        self.C_matrix_Fedorov[i, j] = 2 * self.C(i+1, j+1)
                    else:
                        self.C_matrix_Fedorov[i, j] = np.sqrt(2) * self.C(i+1, j+1)
            self.S_matrix_Fedorov=np.linalg.inv(self.C_matrix_Fedorov)
    
    # read elasticity from Elasticity_T.dat and Elasticity_S.dat
    def read_output(self, fn, args, i):
        """
        Reads output data from a specified file and extracts relevant physical properties.
            
        This method processes the data to extract temperature (T), volume (V), pressure (P),
        density (rho), and various elastic constants. It also computes the compliance matrix
        and initializes the Fedorov matrix if applicable. The method handles different file
        formats and adjusts calculations based on the presence of lattice information.

        Parameters:
        fn (str): The filename from which to read the data.
        args (object): An object containing parameters such as lattice information.
        i (int): The index of the data row to be processed.
    
        Attributes set:
        V (float): Volume of the material.
        M (float): Mass of the material.
        rho (float): Density of the material.
        P (float): Pressure of the material.
        T (float): Temperature of the material.
        system (str): Type of the material system.
        C_matrix (np.array): Elastic stiffness matrix.
        S_matrix (np.array): Elastic compliance matrix.
        BV, GV, EV (float): Bulk, shear, and Young's moduli.
        BR, GR, ER (float): Bulk, shear, and Young's moduli from the compliance matrix.
        BH, GH, EH (float): Hardness-related moduli.
        C_matrix_Fedorov (np.array): Fedorov transformation of the stiffness matrix.
        S_matrix_Fedorov (np.array): Compliance matrix in Fedorov form.
        nuH (float): Poisson's ratio related to hardness.
        kH (float): Shear modulus-to-bulk modulus ratio.
        AVR, AU (float): Volume and Universal anisotropy factors.
        HH (float): Hardness from Chen et al.'s formula.
        CL, CB (float): Longitudinal and bulk sound velocities.
        Debye (float): Debye temperature.

        Raises:
        ValueError: If the input file does not contain the expected data format.
        """
        E_input = np.loadtxt(fn,skiprows=2,dtype=float)
        if len(E_input.shape) == 1:
            E_input = E_input[np.newaxis,:]
        self.T = E_input[i,0]
        self.V = E_input[i,1]
        self.P = E_input[i,2]
        self.rho = E_input[i,3]

        if fn in ['Elasticity_T.dat', 'Elasticity_S.dat']:
            C_format = self.format_Cij(args.lattice, E_input[i,:])
            self.C_matrix = C_format
            self.S_matrix = np.linalg.inv(self.C_matrix)
            self.init_Fedorov_matrix()
            self.BV, self.BR, self.BH, self.GV, self.GR, self.GH, self.EV, self.ER, self.EH, self.nuH, self.kH, self.AVR, self.AU, self.HH, self.CL, self.CB, self.Debye = E_input[i,-17:]
        else:
            if args.lattice != None:
                C_format = self.format_Cij(args.lattice, E_input[i,:])
                self.C_matrix = C_format
                self.S_matrix = np.linalg.inv(self.C_matrix)
                self.init_Fedorov_matrix()
                self.cal_properties(args)
            else:
                self.C_matrix = None
                self.BH, self.GH, self.EH = E_input[i,-3:]
                self.nuH = (1.5 * self.BH - self.GH) / (3. * self.BH + self.GH)
                s = self.nuH
                f = (3*((2*(2/3*(1+s)/(1-2*s))**(3/2))+(1/3*(1+s)/(1-s))**(3/2))**(-1))**(1/3)
                self.Debye = Basic_para().hbar/Basic_para().kB * (6*np.pi**2 * self.V**(1/2))**(1/3)* f * np.sqrt(self.BH/m) * Basic_para().GPa**(1/2) * Basic_para().Ang**(1/2) * Basic_para(). GPa**(1/2) * Basic_para().Ang ** (1/2)

 
    def format_Cij(self, symmetry, E_input):
        """
        Formats the stiffness tensor Cij based on the specified symmetry type and input parameters.
    
        Parameters:
        symmetry (str): The symmetry type of the material. Accepted values are "C", "H", "TI", "TII", 
                        "RI", "RII", "O", "M", and "N".
        E_input (array-like): An array containing the elastic constants required for the specified symmetry.
    
        Returns:
        numpy.ndarray: A 6x6 stiffness tensor Cij formatted according to the specified symmetry.
    
        Raises:
        SystemExit: If the provided symmetry type is not recognized.
        """
        C = np.zeros((6,6))
        if symmetry == "C":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[3,3] = E_input[6]
            C[1,1] = C[0,0]
            C[2,2] = C[0,0]
            C[0,2] = C[0,1]
            C[1,2] = C[0,1]
            C[4,4] = C[3,3]
            C[5,5] = C[3,3]
        elif symmetry == "H":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[2,2] = E_input[7]
            C[3,3] = E_input[8]
            C[1,2] = C[0,2]
            C[1,1] = C[0,0]
            C[4,4] = C[3,3]
            C[5,5] = .5 * (C[0,0] - C[0,1])
        elif symmetry == "TI":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[2,2] = E_input[7]
            C[3,3] = E_input[8]
            C[5,5] = E_input[9]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[4,4] = C[3,3]
        elif symmetry == "TII":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[0,5] = E_input[7]
            C[2,2] = E_input[8]
            C[3,3] = E_input[9]
            C[5,5] = E_input[10]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[1,5] = -C[0,5]
            C[4,4] = C[3,3]
        elif symmetry == "RI":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[0,3] = E_input[7]
            C[2,2] = E_input[8]
            C[3,3] = E_input[9]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[1,3] = -C[0,3]
            C[4,4] = C[3,3]
            C[5,5] = .5 * (C[0,0] - C[0,1])
        elif symmetry == "RII":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[0,3] = E_input[7]
            C[0,4] = E_input[8]
            C[2,2] = E_input[9]
            C[3,3] = E_input[10]
            C[1,1] = C[0,0]
            C[1,2] = C[0,2]
            C[1,3] = -C[0,3]
            C[1,4] = -C[0,4]
            C[3,5] = -C[0,4]
            C[4,4] = C[3,3]
            C[4,5] = C[0,3]
            C[5,5] = .5 * (C[0, 0] - C[0, 1])
        elif symmetry == "O":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[1,1] = E_input[7]
            C[1,2] = E_input[8]
            C[2,2] = E_input[9]
            C[3,3] = E_input[10]
            C[4,4] = E_input[11]
            C[5,5] = E_input[12]
        elif symmetry == "M":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[0,5] = E_input[7]
            C[1,1] = E_input[8]
            C[1,2] = E_input[9]
            C[1,5] = E_input[10]
            C[2,2] = E_input[11]
            C[2,5] = E_input[12]
            C[3,3] = E_input[13]
            C[3,4] = E_input[14]
            C[4,4] = E_input[15]
            C[5,5] = E_input[16]
        elif symmetry == "N":
            C[0,0] = E_input[4]
            C[0,1] = E_input[5]
            C[0,2] = E_input[6]
            C[0,3] = E_input[7]
            C[0,4] = E_input[8]
            C[0,5] = E_input[9]
            C[1,1] = E_input[10]
            C[1,2] = E_input[11]
            C[1,3] = E_input[12]
            C[1,4] = E_input[13]
            C[1,5] = E_input[14]
            C[2,2] = E_input[15]
            C[2,3] = E_input[16]
            C[2,4] = E_input[17]
            C[2,5] = E_input[18]
            C[3,3] = E_input[19]
            C[3,4] = E_input[20]
            C[3,5] = E_input[21]
            C[4,4] = E_input[22]
            C[4,5] = E_input[23]
            C[5,5] = E_input[24]
        else :
            sys.stderr.write('Error: The symmetry is not correct. Please check it!\n')
            os.sys.exit(1)
        for i in range(5):
            for j in range(i+1, 6):
                C[j, i] = C[i, j]
        return C
    
    # calculate the elastic properties based on the elastic stiffness matrix
    def cal_properties(self, args):
        """
        Calculate various material properties based on the elasticity matrix and input parameters.
    
        This method computes the bulk modulus (BV), shear modulus (GV), 
        and their respective reciprocal values (BR, GR), as well as 
        other derived properties such as hardness (HH), sound velocities (CL, CB), 
        and the Debye temperature. The calculations are based on the 
        elasticity matrix (C) and the mass (M) of the material.
    
        Parameters:
        args (object): An object containing the mass of the material (M).

        Attributes set:
        BV, GV, EV (float): Bulk, shear, and Young's moduli.
        BR, GR, ER (float): Bulk, shear, and Young's moduli from the compliance matrix.
        BH, GH, EH (float): Hardness-related moduli.
        nuH (float): Poisson's ratio related to hardness.
        kH (float): Shear modulus-to-bulk modulus ratio.
        AVR, AU (float): Volume and Universal anisotropy factors.
        HH (float): Hardness from Chen et al.'s formula.
        CL, CB (float): Longitudinal and bulk sound velocities.
        Debye (float): Debye temperature.
        """

        C=self.C_matrix
        m = args.M / Basic_para().Na / 1000
        self.BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
        self.GV = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) +
                    3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
       
        S = np.linalg.inv(C)
        self.BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        self.GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 *
            (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
        
        self.BH = 0.50 * (self.BV + self.BR)
        self.GH = 0.50 * (self.GV + self.GR)
        self.EV = 9 * self.BV * self.GV / (3 * self.BV + self.GV)
        self.ER = 9 * self.BR * self.GR / (3 * self.BR + self.GR)
        self.EH = (9. * self.BH * self.GH) / (3. * self.BH + self.GH)
        self.nuH = (1.5 * self.BH - self.GH) / (3. * self.BH + self.GH)
        # kH=GH/BH, from Chen et al.'s paper
        self.kH = self.GH / self.BH 
        self.AVR = (self.GV - self.GR) / (self.GV + self.GR)
        self.AU = 5 * self.GV / self.GR + self.BV / self.BR - 6
        # hardness of material form Chen et al.'s paper
        self.HH = 2 * np.power(self.GH * self.kH **2, 0.585)- 3
        if self.rho == 0:
            self.CL = 0
            self.CB = 0
        else:
            self.CL = pow((self.BH + 4 * self.GH / 3)/self.rho , (1/2))
            self.CB = pow(self.BH/self.rho, (1/2))
        
        s = self.nuH
        f = (3*((2*(2/3*(1+s)/(1-2*s))**(3/2))+(1/3*(1+s)/(1-s))**(3/2))**(-1))**(1/3)
        self.Debye = Basic_para().hbar/Basic_para().kB * (6*np.pi**2 * self.V**(1/2))**(1/3)* f * np.sqrt(self.BH/m) * Basic_para().GPa**(1/2) * Basic_para().Ang**(1/2) * Basic_para(). GPa**(1/2) * Basic_para().Ang ** (1/2)

class Thermal:
    """
    This module defines the Thermal class, which is responsible for managing the thermal properties of materials.

    The Thermal class includes methods to initialize and read thermal properties from an input file. It supports various material symmetries, allowing for the appropriate thermal expansion coefficients to be set based on the provided symmetry type.

    Attributes:
        T (numpy.ndarray): Temperature values.
        V (numpy.ndarray): Volume values.
        P (numpy.ndarray): Pressure values.
        rho (numpy.ndarray): Density values.
        Cp (numpy.ndarray): Specific heat capacity values.
        a (numpy.ndarray): Thermal expansion coefficient a.
        b (numpy.ndarray): Thermal expansion coefficient b.
        c (numpy.ndarray): Thermal expansion coefficient c.

    Methods:
        read_thermal_input(fn, symmetry=None): Reads thermal properties from a specified file and initializes the class attributes based on the data and symmetry type.
    """

    def __init__(self):
        self.T = None
        self.V = None
        self.P = None
        self.rho = None
        self.Cp = None
        self.a = None
        self.b = None
        self.c = None

    # read thermal properties from State_input
    def read_thermal_input(self, fn, symmetry = None):
        """
        Reads thermal input data from a specified file and initializes material properties based on symmetry.
    
        Parameters:
        fn (str): The filename from which to read the thermal input data.
        symmetry (str, optional): The symmetry type of the material, which determines how certain properties are assigned.
            Accepted values include:
            - "C": Cubic symmetry
            - "H": Hexagonal symmetry
            - "TI": Tetragonal I symmetry
            - "TII": Tetragonal II symmetry
            - "RI": Rhombohedral I symmetry
            - "RII": Rhombohedral II symmetry
            - "O": Orthorhombic symmetry
            - "M": Monoclinic symmetry
            - "N": Triclinic symmetry
    
        Raises:
        SystemExit: If the symmetry is not provided or is incorrect, an error message is printed and the program exits.
        """
        thermal_input = np.loadtxt(fn,skiprows=1,dtype=float,encoding='utf-8')
        if len(thermal_input.shape) == 1:
            thermal_input = thermal_input[np.newaxis,:]
        self.T = thermal_input[:,0]
        self.V = thermal_input[:,1]
        self.P = thermal_input[:,2]
        if symmetry != None:
            self.rho = thermal_input[:,3]
            self.Cp = thermal_input[:,4]
            if symmetry == "C":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = self.a
            elif symmetry == "H":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = thermal_input[:,6]
            elif symmetry == "TI":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = thermal_input[:,6]
            elif symmetry == "TII":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = thermal_input[:,6]
            elif symmetry == "RI":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = thermal_input[:,6]
            elif symmetry == "RII":
                self.a = thermal_input[:,5]
                self.b = self.a
                self.c = thermal_input[:,6]
            elif symmetry == "O":
                self.a = thermal_input[:,5]
                self.b = thermal_input[:,6]
                self.c = thermal_input[:,7]
            elif symmetry == "M":
                self.a = thermal_input[:,5]
                self.b = thermal_input[:,6]
                self.c = thermal_input[:,7]
            elif symmetry == "N":
                self.a = thermal_input[:,5]
                self.b = thermal_input[:,6]
                self.c = thermal_input[:,7]
            else:
                sys.stderr.write('Error: The symmetry is not correct. Please check it!\n')
                os.sys.exit(1)
        else:
            sys.stderr.write('Error: The symmetry is not provided. Please provide it first!\n')
            os.sys.exit(1)