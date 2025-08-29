import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps, cm
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import os
import imageio
import sys


class Arrow3D(FancyArrowPatch):
    """
    Arrow3D is a class that extends FancyArrowPatch to create a 3D arrow representation in a Matplotlib plot.

    Attributes:
        _verts3d (tuple): A tuple containing the x, y, and z coordinates of the arrow's vertices.

    Methods:
        draw(renderer): Renders the 3D arrow on the specified renderer.
        do_3d_projection(renderer=None): Projects the 3D coordinates into 2D space and updates the arrow's position.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """
        Draws a 3D arrow on the specified renderer.
    
        This method transforms 3D vertex coordinates into 2D projection coordinates
        and sets the positions for the arrow to be drawn. It utilizes the 
        proj_transform function to handle the projection based on the current axes.
    
        Parameters:
        renderer: The renderer used to draw the arrow.
    
        This method overrides the draw method from the FancyArrowPatch class.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
    
    def do_3d_projection(self, renderer=None):
        """
        Performs a 3D projection of the vertices onto a 2D plane.
    
        This method transforms 3D coordinates (xs3d, ys3d, zs3d) into 2D coordinates 
        using the specified renderer's projection matrix. It sets the positions of 
        the projected points and returns the minimum z-coordinate from the original 
        3D points, which can be useful for depth sorting or visibility determination.
    
        Parameters:
        renderer: Optional; the rendering context used for projection. If None, 
                      defaults to the current axes' projection matrix.
    
        Returns:
        float: The minimum z-coordinate of the original 3D points after projection.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))

        return np.min(zs)

class Anisotropy:
    """
    Anisotropy class for calculating and visualizing various mechanical properties of materials.

    This class provides methods to compute different types of moduli (bulk, Young's, shear, and Poisson's ratio) 
    and sound velocities in three dimensions based on input vectors and material properties. It also includes 
    methods for plotting these properties in 3D and saving the results to files.

    Attributes:
    color_list: A list of colors used for plotting.
    
    Methods:
    dv: Computes the deformation vector for a given input vector.
    nv: Computes the normalized vector for a given input vector.
    mv: Computes the mixed vector for two input vectors.
    vector_to_unit_vector: Converts a vector to its unit vector.
    calc_modulus_3D: Calculates the 3D modulus (bulk, Young's, shear, or Poisson's ratio) based on the provided parameters.
    write_modulus_3D: Writes the computed modulus data to a specified file.
    plot_modulus_3D: Plots the 3D modulus and saves the figure.
    calc_sound_3D: Calculates the 3D sound velocities for the material.
    png_gif: Converts saved PNG figures to GIF format.
    create_gif: Creates a GIF from a list of PNG images.
    """

    def __init__(self):
        self.color_list = ['#00B2CA','#7DCFB6','#FBD1A2','#F79256']
        cm1 = LinearSegmentedColormap.from_list("cm1", self.color_list)
        try:
            colormaps.register(cm1)
        except:
            pass
        try:
            cm.register_cmap(cmap = cm1)
        except:
            pass

    def dv(self, vector):
        """
        Computes the symmetric tensor representation of a given 3D vector.
    
        This method takes a 3-dimensional vector as input and returns a 6-dimensional 
        array representing the components of the symmetric tensor derived from the 
        vector. The output array contains the squares of the vector components and 
        the products of the components, scaled by the square root of 2.
    
        Parameters:
        vector (np.ndarray): A 3-dimensional numpy array representing the vector.
    
        Returns:
        np.ndarray: A 6-dimensional numpy array containing the tensor components.
        """
        d  = np.zeros((6))
        d[0] = vector[0] * vector[0]
        d[1] = vector[1] * vector[1]
        d[2] = vector[2] * vector[2]
        d[3] = np.sqrt(2) * vector[1] * vector[2]
        d[4] = np.sqrt(2) * vector[0] * vector[2]
        d[5] = np.sqrt(2) * vector[0] * vector[1]
        return d

    def nv(self, vector):
        """
        Computes the anisotropic representation of a given vector.
    
        This method takes a 3-dimensional vector as input and returns a 6-dimensional 
        array representing its anisotropic properties. The output array contains the 
        squares of the vector components and the products of the components, scaled 
        by the square root of 2.
    
        Parameters:
        vector (np.ndarray): A 3-dimensional input vector.
    
        Returns:
        np.ndarray: A 6-dimensional array containing the anisotropic representation.
        """
        n  = np.zeros((6))
        n[0] = vector[0] * vector[0]
        n[1] = vector[1] * vector[1]
        n[2] = vector[2] * vector[2]
        n[3] = np.sqrt(2) * vector[1] * vector[2]
        n[4] = np.sqrt(2) * vector[0] * vector[2]
        n[5] = np.sqrt(2) * vector[0] * vector[1]
        return n
    
    def mv(self, vector1, vector2):
        """
        Computes a specific transformation between two vectors.
    
        This method takes two input vectors and calculates a resultant vector 
        based on a defined mathematical operation. The output is a 6-element 
        array where the first three elements are derived from the product of 
        the corresponding elements of the input vectors, and the last three 
        elements are calculated as the sum of products of different pairs of 
        elements from the input vectors.
    
        Parameters:
        vector1 (array-like): The first input vector.
        vector2 (array-like): The second input vector.
    
        Returns:
        numpy.ndarray: A 6-element array representing the transformation result.
        """
        m = np.zeros((6))
        m[0] = np.sqrt(2) * vector1[0]*vector2[0]
        m[1] = np.sqrt(2) * vector1[1]*vector2[1]
        m[2] = np.sqrt(2) * vector1[2]*vector2[2]
        m[3] = vector1[1]*vector2[2] + vector1[2]*vector2[1]
        m[4] = vector1[0]*vector2[2] + vector1[2]*vector2[0]
        m[5] = vector1[0]*vector2[1] + vector1[1]*vector2[0]
        return m

    def vector_to_unit_vector(self,vector):
        """
        Converts a given vector to its corresponding unit vector.
    
        A unit vector is a vector with a magnitude of 1, pointing in the same direction as the original vector.
        This function calculates the magnitude of the input vector and divides the vector by its magnitude to obtain the unit vector.
    
        Parameters:
        vector (numpy.ndarray): The input vector to be converted.
    
        Returns:
        numpy.ndarray: The unit vector in the same direction as the input vector.
        """
        magnitude = np.linalg.norm(vector)
        unit_vector = vector / magnitude
        return unit_vector
    
    # calculate the 3D modulus
    def calc_modulus_3D(self, C, args, thermal_state, n_thermal):
        """
        Calculates various mechanical moduli (bulk, Young's, shear, and Poisson's ratio) in a 3D space based on the provided stiffness matrix and thermal conditions.
        
        This method computes the specified modulus based on the input parameters and saves intermediate results to disk for efficiency. 
        It also generates 3D plots of the computed moduli.

        Parameters:
        C(object) : An object containing material properties, including the stiffness matrix and temperature/pressure conditions.
        args(object) : An object containing parameters for modulus type and plotting options.
        thermal_state(str) : The thermal state of the material, which can be 'isothermal', 'isentropic', or other.
        n_thermal(int) : The number of thermal states to consider for plotting.

        Returns:
        None: The method generates 3D plots of the computed moduli and saves the results to disk.
        """
        n_theta = 181
        n_phi = 91
        n_chi = 361
        phi = np.linspace(0, 1.*np.pi, n_phi)     #z-xy
        theta = np.linspace(0, 2.*np.pi, n_theta) #x-y
        chi = np.linspace(0, 2.*np.pi, n_chi)     #



        script_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_anisotropy_dir = os.path.join(script_dir, 'matrix_anisotropy')
        dv_path = os.path.join(script_dir, 'matrix_anisotropy', 'dv.npy')
        nv_path = os.path.join(script_dir, 'matrix_anisotropy', 'nv.npy')
        mv_path = os.path.join(script_dir, 'matrix_anisotropy', 'mv.npy')
        
        if os.path.exists(matrix_anisotropy_dir) == False:
            os.mkdir(matrix_anisotropy_dir)

        #manager = Manager()
        modulus = args.modulus
        plt_mode = args.plt
        S_matrix = C.S_matrix_Fedorov
        if modulus == 'B':
            if thermal_state == 'isothermal':
                title = 'Isothermal bulk modulus $B$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            elif thermal_state == 'isentropic':
                title = 'Isentropic bulk modulus $B$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            else:
                title = 'Bulk modulus $B$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'

            if os.path.exists(dv_path) == False:
                sys.stdout.write('First time to using this module, it will take a few minutes to perpare some basic inofromation. \n')
                vector_list = [np.array([np.sin(phi[i])*np.cos(theta[j]), np.sin(phi[i])*np.sin(theta[j]), np.cos(phi[i])]) for i in range(n_phi) for j in range(n_theta)]
                vector_list = [self.vector_to_unit_vector(vector) for vector in vector_list]
                dv = [self.dv(vector) for vector in vector_list]
                dv = np.array(dv)
                np.save(dv_path, dv)
            else:
                dv = np.load(dv_path)

            Iv = [[1,1,1,0,0,0] for i in range(n_phi*n_theta)]
            Iv = np.array(Iv)

            M_list = [ 1 / np.dot(np.dot(Iv[i], S_matrix) / 3, dv[i]) for i in range(len(dv))]
            M = np.array(M_list).reshape((n_phi,n_theta))

        elif modulus == 'E':
            if thermal_state == 'isothermal':
                title = 'Isothermal Young\'s modulus $E$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            elif thermal_state == 'isentropic':
                title = 'Isentropic Young\'s modulus $E$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            else:
                title = 'Young\'s modulus $E$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'

            if os.path.exists(dv_path) == False:
                sys.stdout.write('First time to using this module, it will take a few minutes to perpare some basic inofromation. \n')
                vector_list = [np.array([np.sin(phi[i])*np.cos(theta[j]), np.sin(phi[i])*np.sin(theta[j]), np.cos(phi[i])]) for i in range(n_phi) for j in range(n_theta)]
                vector_list = [self.vector_to_unit_vector(vector) for vector in vector_list]
                dv = [self.dv(vector) for vector in vector_list]
                dv = np.array(dv)
                np.save(dv_path, dv)
            else:
                dv = np.load(dv_path)

            M_list = [ 1 / np.dot(np.dot(dv[i], S_matrix), dv[i]) for i in range(len(dv))]
            M = np.array(M_list).reshape((n_phi,n_theta))

        elif modulus == 'G':
            if thermal_state == 'isothermal':
                title = ' isothermal shear modulus $G$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            elif thermal_state == 'isentropic':
                title = ' isentropic shear modulus $G$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            else:
                title = ' shear modulus $G$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            
            if os.path.exists(mv_path) == False:
                sys.stdout.write('First time to using this module, it will take a few minutes to perpare some basic inofromation. \n')
                vector_list1 = [np.array([np.sin(phi[i])*np.cos(theta[j]), np.sin(phi[i])*np.sin(theta[j]), np.cos(phi[i])]) for i in range(n_phi) for j in range(n_theta)]
                vector_list1 = [self.vector_to_unit_vector(vector) for vector in vector_list1]
                vector_list2 = [np.array([np.sin(theta[j])*np.sin(chi[k])-np.cos(phi[i])*np.cos(theta[j])*np.cos(chi[k]),
                                           -np.cos(theta[j])*np.sin(chi[k])-np.cos(phi[i])*np.sin(theta[j])*np.cos(chi[k]),
                                            np.sin(phi[i])*np.cos(chi[k])]) for i in range(n_phi) for j in range(n_theta) for k in range(n_chi)]
                vector_list2 = [self.vector_to_unit_vector(vector) for vector in vector_list2]
                mv = [self.mv(vector_list1[i * n_theta + j], vector_list2[i * n_theta * n_chi + j * n_chi + k]) for i in range(n_phi) for j in range(n_theta) for k in range(n_chi)]
                np.save(mv_path, mv)
            else:
                mv = np.load(mv_path)

            M_list = [ 1 / np.dot(np.dot(mv[i], S_matrix), mv[i]) / 2 for i in range(len(mv))]
            M_list = np.array(M_list).reshape((n_phi,n_theta,n_chi))

            M_average = np.array([np.average(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))
            M_max = np.array([np.max(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))
            M_min = np.array([np.min(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))

        elif modulus == 'nu':
            if thermal_state == 'isothermal':
                title = ' isothermal Poisson ratio $\\nu$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            elif thermal_state == 'isentropic':
                title = ' isentropic Poisson ratio $\\nu$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'
            else:
                title = ' Poisson ratio $\\nu$ \n at '+str(C.T)+'K and '+str(C.P)+' GPa'

            if os.path.exists(dv_path) == False:
                sys.stdout.write('First time to using this module, it will take a few minutes to perpare some basic inofromation. \n')
                vector_list1 = [np.array([np.sin(phi[i])*np.cos(theta[j]), np.sin(phi[i])*np.sin(theta[j]), np.cos(phi[i])]) for i in range(n_phi) for j in range(n_theta)]
                vector_list1 = [self.vector_to_unit_vector(vector) for vector in vector_list1]
                dv = [self.dv(vector) for vector in vector_list1]
                dv = np.array(dv)
                np.save(dv_path, dv)
            else:
                dv = np.load(dv_path)

            if os.path.exists(nv_path) == False:
                sys.stdout.write('First time to using this module, it will take a few minutes to perpare some basic inofromation. \n')
                vector_list2 = [np.array([np.sin(theta[j])*np.sin(chi[k])-np.cos(phi[i])*np.cos(theta[j])*np.cos(chi[k]),
                                           -np.cos(theta[j])*np.sin(chi[k])-np.cos(phi[i])*np.sin(theta[j])*np.cos(chi[k]),
                                            np.sin(phi[i])*np.cos(chi[k])]) for i in range(n_phi) for j in range(n_theta) for k in range(n_chi)]
                vector_list = [self.vector_to_unit_vector(vector) for vector in vector_list2]
                nv = [self.nv(vector) for vector in vector_list2]
                nv = np.array(nv)
                np.save(nv_path, nv)
            else:
                nv = np.load(nv_path)
            E_list = [ 1 / np.dot(np.dot(dv[i], S_matrix), dv[i]) for i in range(len(dv))]
            poisson_list = [ - E_list[i] * np.dot(np.dot(dv[i], S_matrix), nv[i * n_chi + j]) for i in range(len(dv)) for j in range(n_chi)]

            M_list = np.array(poisson_list).reshape((n_phi,n_theta,n_chi))
            M_average = np.array([np.average(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))
            M_max = np.array([np.max(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))
            M_min = np.array([np.min(M_list[i,j,:]) for i in range(n_phi) for j in range(n_theta)]).reshape((n_phi,n_theta))
            
        phi,theta = np.meshgrid(theta,phi)
        if modulus == 'B' or modulus == 'E':
            self.plot_modulus_3D(modulus, M, title, thermal_state, n_thermal, plt_mode)
        if modulus == 'G' or modulus == 'nu':
            self.plot_modulus_3D(modulus+'_average', M_average, "Average"+title, thermal_state, n_thermal, plt_mode)
            self.plot_modulus_3D(modulus+'_max', M_max, "Max"+title, thermal_state, n_thermal, plt_mode)
            self.plot_modulus_3D(modulus+'_min', M_min, "Min"+title, thermal_state, n_thermal, plt_mode)

    def write_modulus_3D(self, data, fn, head, title):
        """
        Writes a 3D modulus data to a specified file.
    
        The function formats the output by aligning the data and headers,
        ensuring that each entry is properly spaced for readability.
    
        Parameters:
        data (list): A 3D list containing the modulus data to be written.
        fn (str): The filename where the data will be saved.
        head (list): A list of header strings to be included in the file.
        title (str): The title to be written at the top of the file.

        Returns:
        None: The 3D modulus data is saved to the specified file.
        """
        fo = open(fn, 'w')
        title = title.replace(' \n ', ' ')
        fo.write(title+'\n')
        for i in range(len(head)):
            fo.write(head[i].ljust(15))
        fo.write('\n')
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data[i][j])):
                    fo.write(str(data[i][j][k]).ljust(14)+' ')
                fo.write('\n')
        fo.close()

    # plot the 3D modulus
    def plot_modulus_3D(self, modulus, M, title, thermal_state, n_thermal, plt_mode):
        """
        Plots a 3D representation of the specified modulus and saves the figure and data.
        
        This function generates a 3D surface plot of the modulus values, applies a color map,
        and annotates the plot with anisotropy information. It also saves the plot and the data
        in specified formats and directories.

        Parameters:
        modulus (str): The type of modulus to plot (e.g., 'v_l', 'B', 'E', etc.).
        M (ndarray): A 2D array representing the modulus values.
        title (str): The title of the plot.
        thermal_state (str): The thermal state ('isothermal' or 'isentropic').
        n_thermal (int): The thermal index for naming the output files.
        plt_mode (str): The format for saving the plot ('png' or 'eps').

        Returns:
        None: The plot and data are saved to the specified directories. 
        """
        n_thermal = n_thermal + 1
        n_theta = 181
        n_phi = 91
        phi = np.linspace(0, 1.*np.pi, n_phi)
        theta = np.linspace(0, 2.*np.pi, n_theta)
        x = np.zeros((n_phi,n_theta))
        y = np.zeros((n_phi,n_theta))
        z = np.zeros((n_phi,n_theta))
        for i in range(0, n_phi):
            for j in range(0, n_theta):
                x[i,j]=np.sin(phi[i])*np.cos(theta[j])*M[i,j]
                y[i,j]=np.sin(phi[i])*np.sin(theta[j])*M[i,j]
                z[i,j]=np.cos(phi[i])*M[i,j]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        if 1<=(M.max()/M.min() <= 1.001):
            N_M = 0.5+(M-M.min())/(M.max()-M.min())/5
        else:
            N_M = (M-M.min())/(M.max()-M.min())
        try:
            cm1 = colormaps['cm1']
        except:
            cm1 = cm.cm1
        surface = ax.plot_surface(
                                x,
                                y,
                                z,
                                alpha=1.0,
                                rstride=1,
                                cstride=1,
                                facecolors=cm1(N_M),
                                linewidth=0,
                                antialiased=False,
                                shade=False)
        ax.grid(False)
        ax.set_axis_off()
        try:
            ax.set_frame_on(False)
        except:
            pass
        ax.set_title(title, fontsize=20, color='black')
        if modulus in ['v_l', 'v_s1', 'v_s2']:
            text_Anisotropy = "Anisotropy is $v_{max}^2/v_{min}^2$="+"{:.3f}".format((M.max()/M.min())**2)
        elif modulus in ['B']:
            text_Anisotropy = "Anisotropy is $B_{max}/B_{min}$="+"{:.3f}".format(M.max()/M.min())
        elif modulus in ['E']:
            text_Anisotropy = "Anisotropy is $E_{max}/E_{min}$="+"{:.3f}".format(M.max()/M.min())
        elif modulus in ['G_average']:
            text_Anisotropy = "Anisotropy is $G_{max}/G_{min}$="+"{:.3f}".format(M.max()/M.min())
        elif modulus in ['G_max']:
            text_Anisotropy = "Anisotropy is $G_{max}/G_{min}$="+"{:.3f}".format(M.max()/M.min())
        elif modulus in ['G_min']:
            text_Anisotropy = "Anisotropy is $G_{max}/G_{min}$="+"{:.3f}".format(M.max()/M.min())
        elif modulus in ['nu_average']:
            text_Anisotropy = "Anisotropy is $\\nu_{max}^2/\\nu_{min}^2$="+"{:.3f}".format((M.max()/M.min())**2)
        elif modulus in ['nu_max']:
            text_Anisotropy = "Anisotropy is $\\nu_{max}^2/\\nu_{min}^2$="+"{:.3f}".format((M.max()/M.min())**2)
        elif modulus in ['nu_min']:
            text_Anisotropy = "Anisotropy is $\\nu_{max}^2/\\nu_{min}^2$="+"{:.3f}".format((M.max()/M.min())**2)
        
        ax.text((M.max()+M.min())/2, (M.max()+M.min())/2, 0, text_Anisotropy, transform=ax.transAxes, color='black', fontsize=16)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
        plt.xlim(-1*M.max(), 1*M.max())
        plt.ylim(-1*M.max(), 1*M.max())
        ax.set_zlim(-1.1*M.max(), 1.1*M.max())
        ax.text(1.5*M.max(), 0, 0, 'x', color='black', fontsize=16, zorder=1)
        ax.text(0, 1.5*M.max(), 0, 'y', color='black', fontsize=16, zorder=1)
        ax.text(0, 0, 1.5*M.max(), 'z', color='black', fontsize=16, zorder=1)
        arrow_prop_dict = dict(mutation_scale=10, arrowstyle="->", shrinkA=0, shrinkB=0)
        a = Arrow3D([-1.5*M.max(), 1.5*M.max()], [0, 0], [0, 0], **arrow_prop_dict, color='black')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [-1.5*M.max(), 1.5*M.max()], [0, 0], **arrow_prop_dict, color='black')
        ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], [-1.8*M.max(), 1.8*M.max()], **arrow_prop_dict, color='black')
        ax.add_artist(a)

        colormap = plt.cm.ScalarMappable(cmap="cm1")
        
        if 1<= M.max()/M.min() <= 1.001:
            colormap.set_clim(M.min()*0.95, M.max()*1.05)
        else:
            colormap.set_array(M)
        try:
            colorbar = plt.colorbar(colormap, fraction=0.046, pad=0.001, shrink=0.6, aspect=10, orientation='horizontal')
        except:
            colorbar = plt.colorbar(colormap, ax=ax, fraction=0.046, pad=0.001, shrink=0.6, aspect=10, orientation='horizontal')

        if modulus in ['v_l', 'v_s1', 'v_s2']:
            colorbar.set_ticks([M.min(), M.max()])
            colorbar.set_ticklabels(["{:.2f} km/s".format(M.min()), "{:.2f} km/s".format(M.max())])
            colorbar.ax.tick_params(labelsize=10)
        elif modulus in ['B', 'E', 'G_average', 'G_max', 'G_min'] and M.max()/M.min() >= 1.001:
            colorbar.set_ticks([M.min(), M.max()])
            colorbar.set_ticklabels(["{:.2f} GPa".format(M.min()), "{:.2f} GPa".format(M.max())])
            colorbar.ax.tick_params(labelsize=10)
        elif modulus in ['B', 'E', 'G_average', 'G_max', 'G_min'] and M.max()/M.min() < 1.001:
            colorbar.set_ticks([M.min()*0.95, (M.min()+M.max())/2, M.max()*1.05])
            colorbar.set_ticklabels(["{:.2f} GPa".format(M.min()*0.95), "{:.2f} GPa".format((M.min()+M.max())/2), "{:.2f} GPa".format(M.max()*1.05)])
            colorbar.ax.tick_params(labelsize=10)
        elif modulus in ['nu_average', 'nu_max', 'nu_min']:
            colorbar.set_ticks([M.min(), M.max()])
            colorbar.set_ticklabels(["{:.6f}".format(M.min()), "{:.6f}".format(M.max())])
            colorbar.ax.tick_params(labelsize=10)
        
        if os.path.exists('figures') == False:
            os.mkdir('figures')
        if os.path.exists('figures/anisotropy') == False:
            os.mkdir('figures/anisotropy')
        
        phi_deg = np.degrees(phi)
        theta_deg = np.degrees(theta)
        data = [ [phi_deg[i], theta_deg[j], M[i,j]] for i in range(n_phi) for j in range(n_theta)]

        if modulus in ['v_l', 'v_s1', 'v_s2']:
            head = ['phi(\u00B0)', 'theta(\u00B0)', modulus+'(km/s)']
        else:
            head = ['phi(\u00B0)', 'theta(\u00B0)', modulus+'(GPa)']
        data = np.array(data).reshape((n_phi, n_theta, 3))

        if thermal_state == 'isothermal':
            fn = 'figures/anisotropy/'+modulus+'_T_'+str(n_thermal)
        elif thermal_state == 'isentropic':
            fn = 'figures/anisotropy/'+modulus+'_S_'+str(n_thermal)
        else:
            fn = 'figures/anisotropy/'+modulus+'_'+str(n_thermal)

        if plt_mode == 'png':
            plt.savefig(fn + '.png', dpi = 300, format='png')
            sys.stdout.write('The figure is saved as '+ fn +'.png\n')
        elif plt_mode == 'eps':
            plt.savefig(fn + '.eps', format='eps')
            sys.stdout.write('The figure is saved as '+ fn +'.eps\n')

        self.write_modulus_3D(data, fn+'.dat', head, title)
        sys.stdout.write('The data is saved as '+ fn +'.dat\n')

        plt.close()

    # calculate the 3D sound velocity
    def calc_sound_3D(self, C, args, thermal_state, n_thermal):
        """
        Calculates the 3D sound velocities (longitudinal and shear) based on the provided elastic constants and thermal state. 
        
        The method computes the Christoffel equation for the 3D elastic constants matrix and
        iterates over a grid of points to calculate the eigenvalues of the elastic constants matrix
        for different angles. It then determines the sound velocities based on the eigenvalues.
    
        Parameters:
        C (object): An object containing the elastic constants matrix (C_matrix) and density (rho).
        args (object): An object containing plotting options (plt).
        thermal_state (str): The thermal state of the material ('isothermal', 'isentropic', or other).
        n_thermal (int): The number of thermal states to consider for plotting.
    
        Returns:
        None: The method generates 3D plots of the sound velocities.
        """
        n_phi = 91
        n_theta = 181
        phi = np.linspace(0, 1.*np.pi, n_phi)     #z-xy
        theta = np.linspace(0, 2.*np.pi, n_theta) #x-y
        Christoffel_equation = np.zeros((3,3))

        Cm = C.C_matrix
        rho = C.rho
        plt_mode = args.plt

        l1  = [np.sin(phi[i])*np.cos(theta[j]) for i in range(n_phi) for j in range(n_theta)]
        l2  = [np.sin(phi[i])*np.sin(theta[j]) for i in range(n_phi) for j in range(n_theta)]
        l3  = [np.cos(phi[i]) for i in range(n_phi) for j in range(n_theta)]
        Christoffel_00 = [Cm[0,0]*l1[i]**2+Cm[5,5]*l2[i]**2+Cm[4,4]*l3[i]**2+2*Cm[4,5]*l2[i]*l3[i]+2*Cm[0,4]*l3[i]*l1[i]+2*Cm[0,5]*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_11 = [Cm[5,5]*l1[i]**2+Cm[1,1]*l2[i]**2+Cm[3,3]*l3[i]**2+2*Cm[1,3]*l2[i]*l3[i]+2*Cm[3,5]*l3[i]*l1[i]+2*Cm[1,5]*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_22 = [Cm[4,4]*l1[i]**2+Cm[3,3]*l2[i]**2+Cm[2,2]*l3[i]**2+2*Cm[2,3]*l2[i]*l3[i]+2*Cm[2,4]*l3[i]*l1[i]+2*Cm[3,4]*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_01 = [Cm[0,5]*l1[i]**2+Cm[1,5]*l2[i]**2+Cm[3,4]*l3[i]**2+(Cm[3,5]+Cm[1,4])*l2[i]*l3[i]+(Cm[0,3]+Cm[4,5])*l3[i]*l1[i]+(Cm[0,1]+Cm[5,5])*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_02 = [Cm[0,4]*l1[i]**2+Cm[3,4]*l2[i]**2+Cm[2,4]*l3[i]**2+(Cm[3,4]+Cm[2,5])*l2[i]*l3[i]+(Cm[0,2]+Cm[4,4])*l3[i]*l1[i]+(Cm[0,3]+Cm[4,5])*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_12 = [Cm[4,5]*l1[i]**2+Cm[1,3]*l2[i]**2+Cm[2,3]*l3[i]**2+(Cm[3,3]+Cm[1,2])*l2[i]*l3[i]+(Cm[2,5]+Cm[3,4])*l3[i]*l1[i]+(Cm[1,4]+Cm[3,5])*l1[i]*l2[i] for i in range(n_phi*n_theta)]
        Christoffel_equation = [np.array([[Christoffel_00[i], Christoffel_01[i], Christoffel_02[i]],
                                 [Christoffel_01[i], Christoffel_11[i], Christoffel_12[i]],
                                 [Christoffel_02[i], Christoffel_12[i], Christoffel_22[i]]]) for i in range(n_phi*n_theta)]
        v_all = [ np.linalg.eigh(Christoffel_equation[i]*(10**9))[0] for i in range(n_phi*n_theta)]
        
        vl = [np.sqrt(v_all[i][2]/rho/1000)/1000 for i in range(n_phi*n_theta)]
        vs1 = [np.sqrt(v_all[i][1]/rho/1000)/1000 for i in range(n_phi*n_theta)]
        vs2 = [np.sqrt(v_all[i][0]/rho/1000)/1000 for i in range(n_phi*n_theta)]
        v_l = np.array(vl).reshape((n_phi,n_theta))
        v_s1 = np.array(vs1).reshape((n_phi,n_theta))
        v_s2 = np.array(vs2).reshape((n_phi,n_theta))

        if thermal_state == 'isothermal':
            title_vl = 'Iisothermal sound velocity $v_l$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs1 = 'Isothermal sound velocity $v_{s1}$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs2 = 'Isothermal sound velocity $v_{s2}$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
        elif thermal_state == 'isentropic':
            title_vl = 'Isentropic sound velocity $v_l$\n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs1 = 'Isentropic sound velocity $v_{s1}$\n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs2 = 'Isentropic sound velocity $v_{s2}$\n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
        else:
            title_vl = 'Sound velocity $v_l$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs1 = 'Sound velocity $v_{s1}$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
            title_vs2 = 'Sound velocity $v_{s2}$ \n at '+'{:.2f}'.format(C.T)+'K and '+'{:.2f}'.format(C.P)+' GPa'
        self.plot_modulus_3D("v_l", v_l, title_vl, thermal_state, n_thermal, plt_mode)
        self.plot_modulus_3D("v_s1", v_s1, title_vs1, thermal_state, n_thermal, plt_mode)
        self.plot_modulus_3D("v_s2", v_s2, title_vs2, thermal_state, n_thermal, plt_mode)
    
    # png to gif
    def png_gif(self, thermal_state):
        """
        Generates GIF images from PNG files based on the specified thermal state.
    
        This method scans the specified directory for PNG files, categorizes them based on the 
        thermal state, sorts them, and then creates GIFs using the sorted lists of images. 
        The output GIF filenames are determined by the thermal state and stored in a dictionary.
        
        Parameters:
        thermal_state (str): The thermal state which can be 'isothermal', 'isentropic', or other.
            - 'isothermal': Processes PNG files prefixed with 'B_T_', 'E_T_', etc.
            - 'isentropic': Processes PNG files prefixed with 'B_S_', 'E_S_', etc.
            - Other: Processes PNG files without thermal state prefixes.
        """
        png_dir = 'figures/anisotropy/'
        png_list = os.listdir(png_dir)
        if thermal_state == 'isothermal':
            name_dic = {'B': 'figures/anisotropy/B_T.gif', 'E': 'figures/anisotropy/E_T.gif', 'G_average': 'figures/anisotropy/G_average_T.gif', 'G_max': 'figures/anisotropy/G_max_T.gif', 'G_min': 'figures/anisotropy/G_min_T.gif', 'nu_average': 'figures/anisotropy/nu_average_T.gif', 'nu_max': 'figures/anisotropy/nu_max_T.gif', 'nu_min': 'figures/anisotropy/nu_min_T.gif', 'v_l': 'figures/anisotropy/v_l_T.gif', 'v_s1': 'figures/anisotropy/v_s1_T.gif', 'v_s2': 'figures/anisotropy/v_s2_T.gif'}
        elif thermal_state == 'isentropic':
            name_dic = {'B': 'figures/anisotropy/B_S.gif', 'E': 'figures/anisotropy/E_S.gif', 'G_average': 'figures/anisotropy/G_average_S.gif', 'G_max': 'figures/anisotropy/G_max_S.gif', 'G_min': 'figures/anisotropy/G_min_S.gif', 'nu_average': 'figures/anisotropy/nu_average_S.gif', 'nu_max': 'figures/anisotropy/nu_max_S.gif', 'nu_min': 'figures/anisotropy/nu_min_S.gif', 'v_l': 'figures/anisotropy/v_l_S.gif', 'v_s1': 'figures/anisotropy/v_s1_S.gif', 'v_s2': 'figures/anisotropy/v_s2_S.gif'}
        else:
            name_dic = {'B': 'figures/anisotropy/B.gif', 'E': 'figures/anisotropy/E.gif', 'G_average': 'figures/anisotropy/G_average.gif', 'G_max': 'figures/anisotropy/G_max.gif', 'G_min': 'figures/anisotropy/G_min.gif', 'nu_average': 'figures/anisotropy/nu_average.gif', 'nu_max': 'figures/anisotropy/nu_max.gif', 'nu_min': 'figures/anisotropy/nu_min.gif', 'v_l': 'figures/anisotropy/v_l.gif', 'v_s1': 'figures/anisotropy/v_s1.gif', 'v_s2': 'figures/anisotropy/v_s2.gif'}
        if thermal_state == 'isothermal':
            B_list = [png for png in png_list if png.startswith('B_T_') and png.endswith('.png')]
            B_list = sorted(B_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            E_list = [png for png in png_list if png.startswith('E_T_') and png.endswith('.png')]
            E_list = sorted(E_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            G_average_list = [png for png in png_list if png.startswith('G_average_T_') and png.endswith('.png')]
            G_average_list = sorted(G_average_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            G_max_list = [png for png in png_list if png.startswith('G_max_T_') and png.endswith('.png')]
            G_max_list = sorted(G_max_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            G_min_list = [png for png in png_list if png.startswith('G_min_T_') and png.endswith('.png')]
            G_min_list = sorted(G_min_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_average_list = [png for png in png_list if png.startswith('nu_average_T_') and png.endswith('.png')]
            nu_average_list = sorted(nu_average_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_max_list = [png for png in png_list if png.startswith('nu_max_T_') and png.endswith('.png')]
            nu_max_list = sorted(nu_max_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_min_list = [png for png in png_list if png.startswith('nu_min_T_') and png.endswith('.png')]
            nu_min_list = sorted(nu_min_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            v_l_list = [png for png in png_list if png.startswith('v_l_T_') and png.endswith('.png')]
            v_l_list = sorted(v_l_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            v_s1_list = [png for png in png_list if png.startswith('v_s1_T_') and png.endswith('.png')]
            v_s1_list = sorted(v_s1_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            v_s2_list = [png for png in png_list if png.startswith('v_s2_T_') and png.endswith('.png')]
            v_s2_list = sorted(v_s2_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
        elif thermal_state == 'isentropic':
            B_list = [png for png in png_list if png.startswith('B_S_') and png.endswith('.png')]
            B_list = sorted(B_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            E_list = [png for png in png_list if png.startswith('E_S_') and png.endswith('.png')]
            E_list = sorted(E_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            G_average_list = [png for png in png_list if png.startswith('G_average_S_') and png.endswith('.png')]
            G_average_list = sorted(G_average_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            G_max_list = [png for png in png_list if png.startswith('G_max_S_') and png.endswith('.png')]
            G_max_list = sorted(G_max_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            G_min_list = [png for png in png_list if png.startswith('G_min_S_') and png.endswith('.png')]
            G_min_list = sorted(G_min_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_average_list = [png for png in png_list if png.startswith('nu_average_S_') and png.endswith('.png')]
            nu_average_list = sorted(nu_average_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_max_list = [png for png in png_list if png.startswith('nu_max_S_') and png.endswith('.png')]
            nu_max_list = sorted(nu_max_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            nu_min_list = [png for png in png_list if png.startswith('nu_min_S_') and png.endswith('.png')]
            nu_min_list = sorted(nu_min_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            v_l_list = [png for png in png_list if png.startswith('v_l_S_') and png.endswith('.png')]
            v_l_list = sorted(v_l_list, key=lambda x: int(x.split('_')[3].split('.')[0]))   
            v_s1_list = [png for png in png_list if png.startswith('v_s1_S_') and png.endswith('.png')]
            v_s1_list = sorted(v_s1_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
            v_s2_list = [png for png in png_list if png.startswith('v_s2_S_') and png.endswith('.png')]
            v_s2_list = sorted(v_s2_list, key=lambda x: int(x.split('_')[3].split('.')[0]))
        else:
            B_list = [png for png in png_list if png.startswith('B_') and not png.startswith('B_T_') and not png.startswith('B_S_') and png.endswith('.png')]
            B_list = sorted(B_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
            E_list = [png for png in png_list if png.startswith('E_') and not png.startswith('E_T_') and not png.startswith('E_S_') and png.endswith('.png')]
            E_list = sorted(E_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
            G_average_list = [png for png in png_list if png.startswith('G_average_') and not png.startswith('G_average_T_') and not png.startswith('G_average_S_') and png.endswith('.png')]
            G_average_list = sorted(G_average_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            G_max_list = [png for png in png_list if png.startswith('G_max_') and not png.startswith('G_max_T_') and not png.startswith('G_max_S_') and png.endswith('.png')]
            G_max_list = sorted(G_max_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            G_min_list = [png for png in png_list if png.startswith('G_min_') and not png.startswith('G_min_T_') and not png.startswith('G_min_S_') and png.endswith('.png')]
            G_min_list = sorted(G_min_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            nu_average_list = [png for png in png_list if png.startswith('nu_average_') and not png.startswith('nu_average_T_') and not png.startswith('nu_average_S_') and png.endswith('.png')]
            nu_average_list = sorted(nu_average_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            nu_max_list = [png for png in png_list if png.startswith('nu_max_') and not png.startswith('nu_max_T_') and not png.startswith('nu_max_S_') and png.endswith('.png')]
            nu_max_list = sorted(nu_max_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            nu_min_list = [png for png in png_list if png.startswith('nu_min_') and not png.startswith('nu_min_T_') and not png.startswith('nu_min_S_') and png.endswith('.png')]
            nu_min_list = sorted(nu_min_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            v_l_list = [png for png in png_list if png.startswith('v_l_') and not png.startswith('v_l_T_') and not png.startswith('v_l_S_') and png.endswith('.png')]
            v_l_list = sorted(v_l_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            v_s1_list = [png for png in png_list if png.startswith('v_s1_') and not png.startswith('v_s1_T_') and not png.startswith('v_s1_S_') and png.endswith('.png')]
            v_s1_list = sorted(v_s1_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
            v_s2_list = [png for png in png_list if png.startswith('v_s2_') and not png.startswith('v_s2_T_') and not png.startswith('v_s2_S_') and png.endswith('.png')]
            v_s2_list = sorted(v_s2_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        final_list = []
        if B_list != []:
            final_list.append('B')
        if E_list != []:
            final_list.append('E')
        if G_average_list != []:
            final_list.append('G_average')
        if G_max_list != []:
            final_list.append('G_max')
        if G_min_list != []:
            final_list.append('G_min')
        if nu_average_list != []:
            final_list.append('nu_average')
        if nu_max_list != []:
            final_list.append('nu_max')
        if nu_min_list != []:
            final_list.append('nu_min')
        if v_l_list != []:
            final_list.append('v_l')
        if v_s1_list != []:
            final_list.append('v_s1')
        if v_s2_list != []:
            final_list.append('v_s2')
        
        for i in final_list:
            if i == 'B':
                image_list = B_list
            elif i == 'E':
                image_list = E_list
            elif i == 'G_average':
                image_list = G_average_list
            elif i == 'G_max':
                image_list = G_max_list
            elif i == 'G_min':
                image_list = G_min_list
            elif i == 'nu_average':
                image_list = nu_average_list
            elif i == 'nu_max':
                image_list = nu_max_list
            elif i == 'nu_min':
                image_list = nu_min_list
            elif i == 'v_l':
                image_list = v_l_list
            elif i == 'v_s1':
                image_list = v_s1_list
            elif i == 'v_s2':
                image_list = v_s2_list
            
            self.create_gif(name_dic[i], image_list, png_dir)

    def create_gif(self, gif_name, image_list, png_dir):
        """
        Creates a GIF from a list of images.
    
        This method takes a list of image file names, reads them from the specified directory,
        and compiles them into a GIF file with the given name. The GIF is saved with a frame rate
        of 2 frames per second and specified encoding options.
    
        Parameters:
        gif_name (str): The name of the output GIF file.
        image_list (list): A list of image file names (without directory path) to be included in the GIF.
        png_dir (str): The directory path where the image files are located.
    
        Returns:
        None: The gif file is saved in the specified directory.
        """
        frame = []
        for j in image_list:
            frame.append(imageio.imread(png_dir+j))
        imageio.mimsave(gif_name, frame, fps = 2, codec='libx264', quality=10,
                        pixelformat = 'yuv444p10', loop = 0)
