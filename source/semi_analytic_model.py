"""
SAM (Semi-Analytic Model) Module

This module implements a semi-analytic model for predicting elastic properties 
of materials under varying pressure and temperature conditions. It provides functionality
for data interpolation, thermodynamic modeling, and visualization of elastic moduli.

Classes:
    SAM: Main class implementing the statistical acoustic model workflow
    
Dependencies:
    numpy, scipy, matplotlib, os, sys
    .parameter.Basic_para (custom module)
    .write_output.write_output (custom module) 
    .elasticity.Elasticity (custom module)
"""

import numpy as np
import scipy
import os
import sys

from matplotlib import cm, colormaps, ticker, rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from .parameter import Basic_para
from .write_output import write_output
from .elasticity import Elasticity

class SAM:
    """
    SAM (Semi-Analytic Model) class for predicting elastic properties.

    This class provides methods for performing piecewise interpolation, calculating 
    elastic characteristic temperatures, and modeling results for elastic modulus 
    based on input parameters. It includes functionality for visualizing results 
    and saving them to disk.

    Attributes:
        input_file (str): Path to input data file.
        T_ref (float): Reference temperature in Kelvin.
        color_list (list): Color palette for visualization.
        weight_coefficient (float): Weighting factor for least squares fitting.
        plot_M_start (float): Start value for colorbar.
        plot_M_end (float): End value for colorbar.
        plot_M_space (float): Spacing for colorbar.
        plot_M_left (float): Left value for colorbar.
        plot_cl_list (list): Contour levels for visualization.
        plot_locations (list): Annotation locations for contour plot.

    Methods:
        interpolate(x, y, x_new): Perform piecewise interpolation using cubic spline or linear.
        theta_E(theta_D): Calculate elastic characteristic temperature.
        delta_theta(theta_E, T): Calculate delta of 1/(exp(theta_E/T) - 1).
        delta_M(M_0, M_T, theta_E, T): Calculate delta of M / delta of 1/(exp(theta_E/T) - 1).
        weight(theta_E, T, delta_M): Calculate weight for least squares fitting.
        isobar_weight(theta_E, T, delta_M): Calculate weight for isobaric conditions.
        model_results(P, T, M0, theta_E, b0, b1): Calculate model results for elastic modulus.
        plot_set_range(M, P_mesh, T_mesh): Set plot range for visualization.
        plot_results(M, P_mesh, T_mesh, modulus, error, state, plt_mode): Visualize and save model results for elastic modulus.
        model_elasticity(C, args, state): Main workflow for elastic properties modeling.
    """
    
    def __init__(self):
        self.input_file = None
        self.T_ref = None
        self.plot_M_start = None
        self.plot_M_end = None
        self.plot_M_space = None
        self.plot_M_left = None
        self.plot_cl_list = None
        self.plot_locations = None
        self.weight_coefficient = None
        self.color_list = ['#00B2CA','#7DCFB6','#FBD1A2','#F79256']
    
    # Interpolation
    def interpolate(self, x, y, x_new):
        """
        Interpolates the given data points (x, y) to estimate new values at specified x_new locations.
    
        This method uses linear interpolation if there are fewer than 4 data points, 
        and cubic spline interpolation for 4 or more data points. The extrapolation 
        option is enabled to handle values outside the range of the input data.
    
        Parameters:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        x_new (array-like): The x-coordinates at which to evaluate the interpolated values.
    
        Returns:
        array: The interpolated y-values corresponding to x_new.
        """
        if len(x) < 4:
            tck = scipy.interpolate.interp1d(x, y, fill_value="extrapolate", bounds_error=False)
            y_new = tck(x_new)
        else:
            tck = scipy.interpolate.CubicSpline(x, y, extrapolate=True)
            y_new = tck(x_new)
        return y_new

    # Elastic characteristic temperature
    def theta_E(self, theta_D):
        """
        Computes the elastic characteristic temperature based on the Debye temperature.
    
        This method applies a specific exponential decay to the Debye temperature to derive the elastic characteristic temperature.
    
        Parameters:
        theta_D (float): The Debye temperature in Kelvin.
    
        Returns:
        float: The calculated elastic characteristic temperature.
        """
        theta_E = theta_D * np.exp(-1.0/3.0)
        return theta_E
    
    # Delta of 1/(exp(theta_E/T) - 1)
    def delta_theta(self, theta_E, T):
        """
        Calculates the difference in the function 1/(exp(theta_E/T) - 1) based on the 
        elastic characteristic temperature and the given temperature.
        
        This method accounts for a reference temperature (T_ref) to adjust the calculation
        of delta_theta. If T_ref is less than or equal to 5, it computes the value directly.
        Otherwise, it computes the difference between two terms involving the elastic 
        characteristic temperature and the reference temperature.
        
        Parameters:
        theta_E (float): Elastic characteristic temperature.
        T (float): Temperature in Kelvin.
        
        Returns:
        float: The computed delta_theta value.
        """
        if self.T_ref <= 5:
            delta_theta = 1 / (np.exp(theta_E/T) - 1)
        else:
            delta_theta = 1 / (np.exp(theta_E/T) - 1) - 1 / (np.exp(theta_E/self.T_ref) -1)
        return delta_theta

    # Delta of M / Delta of 1/(exp(theta_E/T) - 1)
    def delta_M(self, M_0, M_T, theta_E, T):
        """
        Computes the rate of change of the elastic modulus with respect to the 
        change in the characteristic temperature. This function is essential for 
        understanding the material's response to temperature variations.
    
        Parameters:
        M_0 (float): The elastic modulus at the reference state.
        M_T (float): The elastic modulus at the current state.
        theta_E (float): The elastic characteristic temperature.
        T (float): The temperature in Kelvin.
    
        Returns:
        float: The calculated rate of change of the elastic modulus.
        """
        delta_M = M_0 - M_T
        delta_theta = self.delta_theta(theta_E, T)
        delta_M = delta_M / delta_theta
        return delta_M

    # Weight of lstsq
    def weight(self, theta_E, T, delta_M):

        delta_theta = self.delta_theta(theta_E, T)
        weight = delta_M * delta_theta
        return abs(weight) ** self.weight_coefficient

    # Weight of isobar, not used yet
    def isobar_weight(self, theta_E, T, delta_M):
        """
        Calculates the weight for isobaric conditions based on the elastic characteristic temperature, 
        current temperature, and the delta of M. This function is essential for modeling thermodynamic 
        properties in isobaric processes.
        
        Parameters:
        theta_E (float): The elastic characteristic temperature in Kelvin.
        T (float): The current temperature in Kelvin.
        delta_M (float): The delta of M, calculated as delta of 1/(exp(theta_E/T) - 1).
        
        Returns:
        float: The calculated weight for isobaric conditions.
        """ 
        delta_theta = self.delta_theta(theta_E, T)
        weight = delta_M * delta_theta
        return weight
    
    # Results of model
    def model_results(self, P, T, M0, theta_E, b0, b1):
        """
        Calculate the modeled elastic modulus based on pressure and temperature.
        
        This method computes the elastic modulus using the provided pressure and temperature grids,
        along with model parameters and reference elastic modulus. It adjusts the modulus based on
        the difference in characteristic temperature.
        
        Parameters:
        P (ndarray): Pressure grid.
        T (ndarray): Temperature grid.
        M0 (ndarray): Elastic modulus at reference state.
        theta_E (ndarray): Elastic characteristic temperature.
        b0 (float): Model parameter influencing pressure effect.
        b1 (float): Model parameter influencing temperature effect.
        
        Returns:
        ndarray: Modeled elastic modulus as a function of pressure and temperature.
        """
        delta_theta = self.delta_theta(theta_E, T)
        MT = M0 - (b0*P+b1) * delta_theta
        return MT

    # Plot range setting
    def plot_set_range(self, M, P_mesh, T_mesh):
        """
        Determines the range for plotting based on the input mass array (M) and mesh grids for pressure (P_mesh) and temperature (T_mesh).
        
        This method calculates the minimum and maximum values of M, computes the delta, and sets the start and end values for M based on predefined intervals. 
        It also generates a list of contour levels (cl_list) for plotting, ensuring that the levels are appropriately spaced based on the range of M. 
        The method adjusts the starting point if it is too close to the minimum value and handles cases where the contour levels need to be calculated based on the mesh data.
    
        Parameters:
        M (ndarray): A 2D array representing mass values.
        P_mesh (ndarray): A 2D array representing pressure values corresponding to the temperature mesh.
        T_mesh (ndarray): A 2D array representing temperature values.
    
        Attributes set:
        plot_M_start (float): The starting value for M in the plot.
        plot_M_end (float): The ending value for M in the plot.
        plot_M_space (float): The spacing between contour levels.
        plot_M_left (float): The adjustment value for the starting point.
        plot_cl_list (list): The list of contour levels for plotting.
        """
        M_max = max(M.ravel())
        M_min = min(M.ravel())
        delta_M = M_max - M_min
        
        cl_type = None
        if 5 <= delta_M / 1 <= 8:
            M_start = int(M_min / 1 + 1) * 1
            M_end   = int(M_max / 1) * 1
            M_space = 1
            M_left  = 1
        elif 5 <= delta_M / 1.5 <= 8:
            M_start = int(M_min / 1.5 + 1) * 1.5
            M_end   = int(M_max / 1.5) * 1.5
            M_space = 1.5
            M_left  = 1
        elif 5 <= delta_M / 2 <= 8:
            M_start = int(M_min / 2 + 1) * 2
            M_end   = int(M_max / 2) * 2
            M_space = 2
            M_left  = 1
        elif 5 <= delta_M / 2.5 <= 8:
            M_start = int(M_min / 2.5 + 1) * 2.5
            M_end   = int(M_max / 2.5) * 2.5
            M_space = 2.5
            M_left  = 1
        elif 5 <= delta_M / 3 <= 8:
            M_start = int(M_min / 3 + 1) * 3
            M_end   = int(M_max / 3) * 3
            M_space = 3
            M_left  = 1
        elif 5 <= delta_M / 4 <= 8:
            M_start = int(M_min / 4 + 1) * 4
            M_end   = int(M_max / 4) * 4
            M_space = 4
            M_left  = 2
        elif 5 <= delta_M / 5 <= 8:
            M_start = int(M_min / 5 + 1) * 5
            M_end   = int(M_max / 5) * 5
            M_space = 5
            M_left  = 5
        elif 5 <= delta_M / 7.5 <= 8:
            M_start = int(M_min / 7.5 + 1) * 7.5
            M_end   = int(M_max / 7.5) * 7.5
            M_space = 7.5
            M_left  = 3
        elif 5 <= delta_M / 10 <= 8:
            M_start = int(M_min / 10 + 1) * 10
            M_end   = int(M_max / 10) * 10
            M_space = 10
            M_left  = 5
        elif 5 <= delta_M / 15 <= 8:
            M_start = int(M_min / 15 + 1) * 15
            M_end   = int(M_max / 15) * 15
            M_space = 15
            M_left  = 5
        elif 5 <= delta_M / 20 <= 8:
            M_start = int(M_min / 20 + 1) * 20
            M_end   = int(M_max / 20) * 20
            M_space = 20
            M_left   = 10
        elif 5 <= delta_M / 25 <= 8:
            M_start = int(M_min / 25 + 1) * 25
            M_end   = int(M_max / 25) * 25
            M_space = 25
            M_left  = 10
        elif 5 <= delta_M / 30 <= 8:
            M_start = int(M_min / 30 + 1) * 30
            M_end   = int(M_max / 30) * 30
            M_space = 30
            M_left  = 15
        elif 5 <= delta_M / 40 <= 8:
            M_start = int(M_min / 40 + 1) * 40
            M_end   = int(M_max / 40) * 40
            M_space = 40
            M_left  = 20
        elif 5 <= delta_M / 50 <= 8:
            M_start = int(M_min / 50 + 1) * 50
            M_end   = int(M_max / 50) * 50
            M_space = 50
            M_left  = 20
        elif 5 <= delta_M / 60 <= 8:
            M_start = int(M_min / 60 + 1) * 60
            M_end   = int(M_max / 60) * 60
            M_space = 60
            M_left  = 30
        elif 5 <= delta_M / 80 <= 8:
            M_start = int(M_min / 80 + 1) * 80
            M_end   = int(M_max / 80) * 80
            M_space = 80
            M_left  = 40
        elif 5 <= delta_M / 100 <= 8:
            M_start = int(M_min / 100 + 1) * 100
            M_end   = int(M_max / 100) * 100
            M_space = 100
            M_left  = 50
        elif 5 <= delta_M / 150 <= 8:
            M_start = int(M_min / 150 + 1) * 150
            M_end   = int(M_max / 150) * 150
            M_space = 150
            M_left  = 50
        elif 5 <= delta_M / 200 <= 8:
            M_start = int(M_min / 200 + 1) * 200
            M_end   = int(M_max / 200) * 200
            M_space = 200
            M_left  = 100
        elif 5 <= delta_M / 250 <= 8:
            M_start = int(M_min / 250 + 1) * 250
            M_end   = int(M_max / 250) * 250
            M_space = 250
            M_left  = 100
        elif 5 <= delta_M / 300 <= 8:
            M_start = int(M_min / 300 + 1) * 300
            M_end   = int(M_max / 300) * 300
            M_space = 300
            M_left  = 100
        elif 5 <= delta_M / 400 <= 8:
            M_start = int(M_min / 400 + 1) * 400
            M_end   = int(M_max / 400) * 400
            M_space = 400
            M_left  = 200
        elif 5 <= delta_M / 500 <= 8:
            M_start = int(M_min / 500 + 1) * 500
            M_end   = int(M_max / 500) * 500
            M_space = 500
            M_left  = 200
        elif 5 <= delta_M / 600 <= 8:
            M_start = int(M_min / 600 + 1) * 600
            M_end   = int(M_max / 600) * 600
            M_space = 600
            M_left  = 300
        elif 5 <= delta_M / 800 <= 8:
            M_start = int(M_min / 800 + 1) * 800
            M_end   = int(M_max / 800) * 800
            M_space = 800
            M_left  = 400
        elif 5 <= delta_M / 1000 <= 8:
            M_start = int(M_min / 1000 + 1) * 1000
            M_end   = int(M_max / 1000) * 1000
            M_space = 1000
            M_left  = 500
        elif 5 <= delta_M / 1500 <= 8:
            M_start = int(M_min / 1500 + 1) * 1500
            M_end   = int(M_max / 1500) * 1500
            M_space = 1500
            M_left  = 500
        elif 5 <= delta_M / 2000 <= 8:
            M_start = int(M_min / 2000 + 1) * 2000
            M_end   = int(M_max / 2000) * 2000
            M_space = 2000
            M_left  = 1000
        elif 5 <= delta_M / 2500 <= 8:
            M_start = int(M_min / 2500 + 1) * 2500
            M_end   = int(M_max / 2500) * 2500
            M_space = 2500
            M_left  = 1000
        elif 5 <= delta_M / 3000 <= 8:
            M_start = int(M_min / 3000 + 1) * 3000
            M_end   = int(M_max / 3000) * 3000
            M_space = 3000
            M_left  = 1000
        elif 5 <= delta_M / 4000 <= 8:
            M_start = int(M_min / 4000 + 1) * 4000
            M_end   = int(M_max / 4000) * 4000
            M_space = 4000
            M_left  = 2000
        elif 5 <= delta_M / 5000 <= 8:
            M_start = int(M_min / 5000 + 1) * 5000
            M_end   = int(M_max / 5000) * 5000
            M_space = 5000
            M_left  = 2000
        else:
            M_start = M_min
            M_end = M_max
            M_space = (M_end - M_start)/10
            M_left = 0
            cl_type = 'auto'

        M_num = int((M_end - M_start) / M_space)
        if abs(M_start - M_min) < M_space/2:
            M_start = M_start + M_left
                
        if M_max - M_start - M_num * M_space  > M_left and cl_type == None :
            cl_list = [int(M_min)] + [M_start + M_space * i for i in range(M_num+1)] + [int(M_max)]
        elif cl_type == None:
            cl_list = [int(M_min)] + [M_start + M_space * i for i in range(M_num)] + [int(M_max)]
        else:
            cl_list =None
        locations = []
        """
        for cl_value in cl_list:
            if cl_value != M_max and cl_value != M_min:
                temp = []
                for T_temp in np.unique(T_mesh):
                    P_value = P_mesh[T_mesh == T_temp]
                    M_value = M[T_mesh == T_temp]
                    d_M = (M_value - cl_value) / cl_value
                    if np.all(d_M > 0) or np.all(d_M < 0):
                        continue
                    for i in range(len(d_M)-1):
                        if d_M[i] * d_M[i+1] < 0:
                            P_temp = (P_value[i] + P_value[i+1])/2
                            temp.append([P_temp, T_temp])
                
                if temp != [] and abs(temp[0][1] - temp[-1][1]) >  abs(T_mesh.max()-T_mesh.min())/5:
                    P_mid = (temp[0][0] + temp[-1][0])/2
                    T_mid = (temp[0][1] + temp[-1][1])/2
                    location = (P_mid, T_mid)
                    locations.append(location)
                elif temp != [] and abs(temp[0][0]-temp[-1][0]) > abs(P_mesh.max()-P_mesh.min())/5:
                    P_mid = (temp[0][0] + temp[-1][0])/2
                    T_mid = (temp[0][1] + temp[-1][1])/2
                    location = (P_mid, T_mid)
                    locations.append(location)
        """
        self.plot_M_start = M_start
        self.plot_M_end = M_end
        self.plot_M_space = M_space
        self.plot_M_left = M_left
        self.plot_cl_list = cl_list
        #self.plot_locations = locations

    # Plot results
    def plot_results(self, M, P_mesh, T_mesh, modulus, error, state, plt_mode):
        """
        Plots the results of the semi-analytic model.
    
        This method generates a contour plot of the modeled data based on the provided parameters.
        It customizes the plot's appearance, including tick parameters, spine widths, and color normalization.
        The function also saves the plot in the specified format (PNG or EPS) and outputs the file name.
    
        Parameters:
        M (ndarray): The matrix of modeled values to be plotted.
        P_mesh (ndarray): The mesh grid for pressure values.
        T_mesh (ndarray): The mesh grid for temperature values.
        modulus (str): The type of modulus being modeled (e.g., 'V' for volume).
        error (float): The root mean square error of the model.
        state (str): The state of the model ('isothermal' or 'isentropic').
        plt_mode (str): The format to save the plot ('png' or 'eps').
    
        Returns:
        None: The function is to generate and save the plot.
        """
        
        plt.tick_params(direction='in', length=6, width=2)
        ax=plt.gca()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        for spine in ax.spines.values():
            spine.set_linewidth(3)      
        norm = plt.Normalize(min(M.ravel()), max(M.ravel()))
        plt.tick_params(direction='in', length=6, width=2)
        ax=plt.gca()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        norm = plt.Normalize(min(M.ravel()), max(M.ravel()))
        plt.contourf(P_mesh, T_mesh, M, 100, cmap='cm1', norm=norm)
        cl = None
        cl = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=8)
        #if self.plot_cl_list != None:
        #    contour = plt.contour(P_mesh, T_mesh, M, self.plot_cl_list, norm=norm, colors = 'black', linestyles= 'dashed', linewidths=2)
        #else:
        #    contour = plt.contour(P_mesh, T_mesh, M, 8, colors = 'black', linestyles= 'dashed', linewidths=2)
        contour = plt.contour(P_mesh, T_mesh, M, 8, colors = 'black', linestyles= 'dashed', linewidths=2)
        labels=plt.clabel(contour, inline=True, fontsize=18) #, manual=self.plot_locations)
        plt.xlabel('${P}$ (GPa)', fontsize=18, color='black')
        plt.ylabel('${T}$ (K)', fontsize=18, color='black')
        if modulus != 'V':
            cl.set_label('${M}$ (GPa)', fontsize=18, color='black')
        else:
            cl.set_label('${V}$ (ang$^3$/atom)', fontsize=18, color='black')

        tick_locator = ticker.MaxNLocator(nbins=8)
        cl.locator = tick_locator
        #plot_cl_list = self.plot_cl_list
        #cl.set_ticks(plot_cl_list)
        #cl.update_ticks()
        if modulus != 'V':
            title = '${'+modulus+'}$ is modeled with RMSE: ' + "{:.2f}".format(error) + ' GPa'
        else:
            title = '${'+modulus+'}$ is modeled with RMSE: ' + "{:.4f}".format(error) + ' ang$^3$/atom'
        plt.title(title, fontsize=18, color='black')
        if state == 'isothermal':
            fig_name = 'figures/model/isothermal_'+modulus+'_model'
        elif state == 'isentropic':
            fig_name = 'figures/model/isentropic_'+modulus+'_model'
        else:
            fig_name = 'figures/model/'+modulus+'_model'
        if plt_mode == 'png':
            fig_name = fig_name + '.png'
            plt.savefig(fig_name, dpi = 600, format='png')
        elif plt_mode == 'eps':
            fig_name = fig_name + '.eps'
            plt.savefig(fig_name, format='eps')
        sys.stdout.write('The figure of modeled ' + modulus +' is saved as '+fig_name+'\n')
        plt.close()

    # Model of elasticity
    def model_elasticity(self, C, args, state):
        """
        Calculates the elasticity model parameters based on the provided material properties and state conditions.
    
        This method processes a list of material constants, computes reference states, and interpolates 
        values for thermal states. It generates model results for various elastic constants and moduli, 
        and saves the results to specified output files. The method supports different lattice types and 
        allows for customizable pressure and temperature ranges.
    
        Parameters:
        C (list): A list of material constants for different states.
        args (object): An object containing various parameters including weight, temperature reference, 
                       pressure range, and temperature range.
        state (str): The state of the model, which can be 'isothermal', 'isentropic', or other.
    
        Returns:
        None: The results are saved to files.
        """
        num = len(C)
        self.weight_coefficient = args.weight
        T_list = [C[i].T for i in range(num)]
        P_list = [C[i].P for i in range(num)]
        V_list = [C[i].V for i in range(num)]
        Debye_list = [C[i].Debye for i in range(num)]

        B_list = [C[i].BH for i in range(num)]
        G_list = [C[i].GH for i in range(num)]
        E_list = [C[i].EH for i in range(num)]

        modulus_list = ['B', 'G', 'E']

        Cij_indices = {'C11': (1,1), 'C12': (1,2), 'C13': (1,3), 'C14': (1,4), 'C15': (1,5), 'C16': (1,6), 'C22': (2,2), 'C23': (2,3), 'C24': (2,4), 'C25': (2,5), 'C26': (2,6), 'C33': (3,3), 'C34': (3,4), 'C35': (3,5), 'C36': (3,6), 'C44': (4,4), 'C45': (4,5), 'C46': (4,6), 'C55': (5,5), 'C56': (5,6), 'C66': (6,6)}
        if args.lattice == 'C':
            Cij_list = ['C11', 'C12', 'C44']
        elif args.lattice == 'H':
            Cij_list = ['C11', 'C12', 'C13', 'C33', 'C44']
        elif args.lattice == 'T':
            Cij_list = ['C11', 'C12', 'C13', 'C33', 'C44', 'C66']
        elif args.lattice == 'TII':
            Cij_list = ['C11', 'C12', 'C13', 'C33', 'C44', 'C66', 'C16']
        elif args.lattice == 'RI':
            Cij_list = ['C11', 'C12', 'C13', 'C14', 'C33', 'C44']
        elif args.lattice == 'RII':
            Cij_list = ['C11', 'C12', 'C13', 'C14', 'C15', 'C33', 'C44']
        elif args.lattice == 'O':
            Cij_list = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33', 'C44', 'C55', 'C66']
        elif args.lattice == 'M':
            Cij_list = ['C11', 'C12', 'C13', 'C16', 'C22', 'C23', 'C26', 'C33', 'C36', 'C44', 'C45', 'C55', 'C66']
        elif args.lattice == 'N':
            Cij_list = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C22', 'C23', 'C24', 'C25', 'C26', 'C33', 'C34', 'C35', 'C36', 'C44', 'C45', 'C46', 'C55', 'C56', 'C66']
        
        M_list  = ['V'] + Cij_list + modulus_list

        Cij_dict = {Cij: [] for Cij in Cij_list}
        for i in range(num):
            for Cij in Cij_list:
                indices = Cij_indices[Cij]
                Cij_dict[Cij].append(C[i].C(*indices))

        self.T_ref = args.T_ref
        ref_state_list = [i for i in range(num) if T_list[i] == self.T_ref]

        thermal_state_list = [i for i in range(num) if T_list[i] != self.T_ref]

        ref_P_list = [P_list[i] for i in ref_state_list]
        sorted_ref_state_list = [x for _, x in sorted(zip(ref_P_list, ref_state_list))]
        P_ref = [P_list[i] for i in sorted_ref_state_list]
        V_ref = [V_list[i] for i in sorted_ref_state_list]
        Debye_ref = [Debye_list[i] for i in sorted_ref_state_list]

        B_ref = [B_list[i] for i in sorted_ref_state_list]
        G_ref = [G_list[i] for i in sorted_ref_state_list]
        E_ref = [E_list[i] for i in sorted_ref_state_list]
        

        Cij_dict_ref = {Cij: [] for Cij in Cij_list}
        for i in sorted_ref_state_list:
            for Cij in Cij_list:
                indices = Cij_indices[Cij]
                Cij_dict_ref[Cij].append(C[i].C(*indices))
        
        P_list_thermal = np.array([P_list[i] for i in thermal_state_list])
        T_list_thermal = np.array([T_list[i] for i in thermal_state_list])
        Debye_ref_thermal = self.interpolate(P_ref, Debye_ref, P_list_thermal)

        M_dict_ref_thermal = {M: [] for M in M_list}
        delta_dict = {M: [] for M in M_list}
        b0_dict = {M: [] for M in M_list}
        b1_dict = {M: [] for M in M_list}
        error_dict = {M: [] for M in M_list}
        M_fit_dict = {M: [] for M in M_list}

        cm1 = LinearSegmentedColormap.from_list("cm1", self.color_list)
        try:
            colormaps.register(cm1)
        except:
            cm.register_cmap(cmap = cm1)
        
        if os.path.exists('figures') == False:
            os.mkdir('figures')
        if os.path.exists('figures/model') == False:
            os.mkdir('figures/model')

        if args.P_range != None:
            P_range = np.linspace(args.P_range[0], args.P_range[1], args.P_range[2])
        else:
            P_range = np.linspace(min(P_list), max(P_list), 101)
        if args.T_range != None:
            T_range = np.linspace(args.T_range[0], args.T_range[1], args.T_range[2])
        else:
            T_range = np.linspace(min(T_list), max(T_list), 101)
        P_mesh, T_mesh = np.meshgrid(P_range, T_range)

        for M in M_list:
            if M in ['B', 'G', 'E', 'V']:
                M_ref = locals()[M+'_ref']
                M_thermal = locals()[M+'_list']
            else:
                M_ref = Cij_dict_ref[M]
                M_thermal = Cij_dict[M]
            M_dict_ref_thermal[M] = self.interpolate(P_ref, M_ref, P_list_thermal)
            M_thermal = [M_thermal[i] for i in thermal_state_list]
            delta_dict[M] = self.delta_M(M_dict_ref_thermal[M], M_thermal, self.theta_E(Debye_ref_thermal), T_list_thermal)
            weight = self.weight(self.theta_E(Debye_ref_thermal), T_list_thermal, delta_dict[M])
            P_list_thermal_stack = np.vstack((P_list_thermal, np.ones(len(P_list_thermal)))).T * np.sqrt(weight[:, np.newaxis])
            delta_M_weighted = delta_dict[M] * np.sqrt(weight)
            b0_dict[M], b1_dict[M] = np.linalg.lstsq(P_list_thermal_stack, delta_M_weighted, rcond = -1)[0]
            error_dict[M] = np.sqrt(np.sum((delta_dict[M] - (b0_dict[M]*P_list_thermal + b1_dict[M]))**2) / len(P_list_thermal))
            M0_mesh = self.interpolate(P_ref, M_ref, P_mesh)
            theta_E_mesh = self.theta_E(self.interpolate(P_ref, Debye_ref, P_mesh))
            M_fit_dict[M] = self.model_results(P_mesh, T_mesh, M0_mesh, theta_E_mesh, b0_dict[M], b1_dict[M])
            
            self.plot_set_range(M_fit_dict[M], P_mesh, T_mesh)
            self.plot_results(M_fit_dict[M], P_mesh, T_mesh, M, error_dict[M], state, args.plt)
        if state == 'isothermal':
            fn1 = 'figures/model/Elasticity_T_model.dat'
            fn2 = 'figures/model/Elasticity_T_model_calc.dat'
        elif state == 'isentropic':
            fn1 = 'figures/model/Elasticity_S_model.dat'
            fn2 = 'figures/model/Elasticity_S_model_calc.dat'
        else:
            fn1 = 'figures/model/Elasticity_model.dat'
            fn2 = 'figures/model/Elasticity_model_calc.dat'
        fo1 = open(fn1, 'w')

        if state == 'isothermal':
            fn = 'Elasticity_T.dat'
        elif state == 'isentropic': 
            fn = 'Elasticity_S.dat'
        elif state == 'other':
            fn = args.read_file

        write_model = write_output()
        format_str = 'The SAM model uses the reference state from {} at T = {:<14.6f} K\n'
        fo1.write(format_str.format(fn, self.T_ref))
        write_model.filehead_model_ref(args.lattice, fo1)
        for i in ref_state_list:
            write_model.Cinf_model_ref(args.lattice, fo1, C[i])
        fo1.write('The SAM model parameters are as follows:\n')
        for i in M_list:
            if i == 'V':
                format_str = '{}: b0 is {:<10.8f} (ang$^3/atom)     b1 is {:<10.8f} (ang^3/atom/GPa)     RMSE is {:<8.6} (ang^3/atom)\n'
            else:
                format_str = '{}: b0 is {:<8.6f} (GPa)     b1 is {:<8.6f}     RMSE is {:<8.6f} (GPa)\n'
            fo1.write(format_str.format(i, b0_dict[i], b1_dict[i], error_dict[i]))

        fo1.write('The modeled elastic constants and moduli are as follows:\n')
        head = ['T(K)', 'P(GPa)', 'V(ang^3/atom)'] + Cij_list + ['B', 'G', 'E']
        for i in head:
            if i in Cij_list+['B', 'G', 'E']:
                temp = i+'(GPa)'
                fo1.write(temp.ljust(15))
            else:
                fo1.write(i.ljust(15))
        fo1.write('\n')

        for j in range(args.T_range[2]):
            for i in range(args.P_range[2]):
                format_str = '{:<14.4f} '* (6+len(Cij_list))
                temp = [T_mesh[j][i], P_mesh[j][i], M_fit_dict['V'][j][i]] + [M_fit_dict[Cij][j][i] for Cij in Cij_list] + [M_fit_dict[M][j][i] for M in ['B', 'G', 'E']]
                fo1.write(format_str.format(*temp))
                fo1.write('\n') 
        fo1.close()
        sys.stdout.write('The modeled elastic constants and moduli is saved as '+fn1+'\n')

        write_cal = write_output()
        write_cal.filehead(args.lattice, fn2)
        for j in range(args.T_range[2]):
            for i in range(args.P_range[2]):
                line = Elasticity()
                line.T = T_mesh[j][i]
                line.P = P_mesh[j][i]
                line.V = M_fit_dict['V'][j][i]
                line.rho = args.M/(line.V * Basic_para().Na/1E24)
                temp = [line.T, line.V, line.P, line.rho]+[M_fit_dict[Cij][j][i] for Cij in Cij_list]
                line.C_matrix = line.format_Cij(args.lattice, temp)
                line.cal_properties(args)
                write_cal.Cinf(args.lattice, fn2, line)
        
        sys.stdout.write('Elasticity calculated based on modeled C matrix is saved as '+fn2+'\n')