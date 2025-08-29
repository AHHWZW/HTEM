class write_output:
    """
    This module defines the `write_output` class, which is responsible for writing formatted output data related to various crystal structures and their elastic properties to specified output files.

    Attributes:
        para: A dictionary of parameters with their corresponding 
        units, used to format the output data.

    Methods:
        filehead(LC, output_file): Writes the header for the output file based on the crystal structure type.
        filehead_model_ref(LC, fo): Writes the reference header for the output file.
        Cinf(LC, output_file, E): Appends elasticity data to the output file.
        Cinf_model_ref(LC, fo, E): Appends reference elasticity data to the output file.
    """
    
    # write the file head for the output file
    def __init__(self):
        self.para = {'T':'T(K)','P':'P(GPa)','C11':'C11(GPa)','C12':'C12(GPa)','C13':'C13(GPa)','C14':'C14(GPa)','C15':'C15(GPa)','C16':'C16(GPa)',
                'C22':'C22(GPa)','C23':'C23(GPa)','C24':'C24(GPa)','C25':'C25(GPa)','C26':'C26(GPa)','C33':'C33(GPa)','C34':'C34(GPa)','C35':'C35(GPa)',
                'C36':'C36(GPa)','C44':'C44(GPa)','C45':'C45(GPa)','C46':'C46(GPa)','C55':'C55(GPa)','C56':'C56(GPa)','C66':'C66(GPa)',
                'BV':'BV(GPa)','BR':'BR(GPa)','BH':'BH(GPa)','GV':'GV(GPa)','GR':'GR(GPa)','GH':'GH(GPa)', 'EV':'EV(GPa)', 'ER':'ER(GPa)','EH':'EH(GPa)',
                'nuH':'nuH','kH':'kH', 'HH':'HH(GPa)','AVR':'AVR','AU':'AU','CL':'CL(km/s)','CB':'CB(km/s)','V':'V(ang^3/atom)','Debye':'Debye(K)','rho':'rho(g/cm^3)'}

    def filehead(self, LC, output_file):
        """
        Writes the header information for different crystal structures to the specified output file.
    
        The function writes a header line indicating the type of structure and the corresponding space-group number range,
        followed by the formatted parameters related to the structure, each left-justified to a width of 15 characters.
    
        Parameters:
        LC (str): A character representing the type of crystal structure. 
                  Possible values include:
                  - 'C': Cubic
                  - 'H': Hexagonal
                  - 'TI': Tetragonal I
                  - 'TII': Tetragonal II
                  - 'RI': Rhombohedral I
                  - 'RII': Rhombohedral II
                  - 'O': Orthorhombic
                  - 'M': Monoclinic
                  - 'N': Triclinic
        output_file (str): The path to the file where the output data will be written.

        Returns:
        None: The file header is written to the specified output file.
        """
        fo = open(output_file,'w')
        para = self.para
        if (LC == 'C'):
            fo.write("Output data for Cubic structure, Space-Group Number between 195 and 230: \n")
            for i in ['T','V','P','rho','C11','C12','C44','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'H'):
            fo.write("Output data for Hexagonal structure, Space-Group Number between 168 and 194: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C33','C44','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'TI'):
            fo.write("Output data for Tetragonal I structure, Space-Group Number between 89 and 142: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C33','C44','C66','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'TII'):
            fo.write("Output data for Tetragonal II structure, Space-Group Number between 75 and 88: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C16','C33','C44','C66','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'RI'):
            fo.write("Output data for Rhombohedral I structure, Space-Group Number between 149 and 167: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C33','C44','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'RII'):
            fo.write("Output data for Rhombohedral II structure, Space-Group Number between 143 and 148: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C15','C33','C44','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'O'):
            fo.write("Output data for Orthorhombic structure, Space-Group Number between 16 and 74: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C22','C23','C33','C44','C55','C66','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'M'):
            fo.write("Output data for Monoclinic structure, Space-Group Number between 3 and 15: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C16','C22','C23','C26','C33','C36','C44','C45','C55','C66','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'N'):
            fo.write("Output data for Triclinic structure, Space-Group Number between 1 and 2: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C15','C16','C22','C23','C24','C25','C26','C33','C34','C35','C36','C44','C45','C46','C55','C56','C66','BV','BR','BH','GV','GR','GH','EV','ER','EH','nuH','kH','HH','AVR','AU','CL','CB','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        fo.close()

    def filehead_model_ref(self, LC, fo):
        """
        Writes the header information for different crystal structures to the specified output file for reference state.
    
        The function writes a header line indicating the type of structure and the corresponding space-group number range,
        followed by the formatted parameters related to the structure, each left-justified to a width of 15 characters.
    
        Parameters:
        LC (str): A character representing the type of crystal structure. 
                  Possible values include:
                  - 'C': Cubic
                  - 'H': Hexagonal
                  - 'TI': Tetragonal I
                  - 'TII': Tetragonal II
                  - 'RI': Rhombohedral I
                  - 'RII': Rhombohedral II
                  - 'O': Orthorhombic
                  - 'M': Monoclinic
                  - 'N': Triclinic
        output_file (str): The path to the file where the output data will be written.

        Returns:
        None: The file header is written to the specified output file.
        """
        para = self.para
        if (LC == 'C'):
            fo.write("Reference data for Cubic structure, Space-Group Number between 195 and 230: \n")
            for i in ['T','V','P','rho','C11','C12','C44','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'H'):
            fo.write("Reference data for Hexagonal structure, Space-Group Number between 168 and 194: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C33','C44','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'TI'):
            fo.write("Reference data for Tetragonal I structure, Space-Group Number between 89 and 142: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C33','C44','C66','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'TII'):
            fo.write("Reference data for Tetragonal II structure, Space-Group Number between 75 and 88: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C16','C33','C44','C66','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'RI'):
            fo.write("Reference data for Rhombohedral I structure, Space-Group Number between 149 and 167: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C33','C44','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'RII'):
            fo.write("Reference data for Rhombohedral II structure, Space-Group Number between 143 and 148: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C15','C33','C44','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'O'):
            fo.write("Reference data for Orthorhombic structure, Space-Group Number between 16 and 74: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C22','C23','C33','C44','C55','C66','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'M'):
            fo.write("Reference data for Monoclinic structure, Space-Group Number between 3 and 15: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C16','C22','C23','C26','C33','C36','C44','C45','C55','C66','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
        elif (LC == 'N'):
            fo.write("Reference data for Triclinic structure, Space-Group Number between 1 and 2: \n")
            for i in ['T','V','P','rho','C11','C12','C13','C14','C15','C16','C22','C23','C24','C25','C26','C33','C34','C35','C36','C44','C45','C46','C55','C56','C66','BH','GH','EH','Debye']:
                fo.write(para[i].ljust(15))
            fo.write('\n')
    
    # write the output data for the elasticity
    def Cinf(self, LC='C', output_file = 'Elasticity_T.dat', E=None):
        """
        Writes material properties and compliance matrix values to an output file.
            
        The function appends formatted values of the compliance matrix and other material properties
        to the specified output file based on the selected compliance type (LC).
    
        Parameters:
        LC (str): A character representing the type of crystal structure. 
                  Possible values include:
                  - 'C': Cubic
                  - 'H': Hexagonal
                  - 'TI': Tetragonal I
                  - 'TII': Tetragonal II
                  - 'RI': Rhombohedral I
                  - 'RII': Rhombohedral II
                  - 'O': Orthorhombic
                  - 'M': Monoclinic
                  - 'N': Triclinic
        output_file (str): The name of the file to which the data will be appended. Default is 'Elasticity_T.dat'.
        E (object): An object containing material properties and compliance matrix data.

        Returns:
        None: The formatted data is appended to the specified output file.
        """
        C = E.C_matrix
        fo = open(output_file,'a')
        format_str = '{:<14.4f} '* 4
        fo.write(format_str.format(E.T,E.V,E.P,E.rho))

        if (LC == 'C'):
            format_str = '{:<14.4f} '* 3 
            fo.write(format_str.format(C[0,0],C[0,1],C[3,3]))
        elif (LC == 'H'):
            format_str = '{:<14.4f} '* 5
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[2,2],C[3,3]))
        elif (LC == 'TI'):
            format_str = '{:<14.4f} '* 6
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[2,2],C[3,3],C[5,5]))
        elif (LC == 'TII'):
            format_str = '{:<14.4f} '* 7
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,5],C[2,2],C[3,3],C[5,5]))
        elif (LC == 'RI'):
            format_str = '{:<14.4f} '* 6
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[2,2],C[3,3]))
        elif (LC == 'RII'):
            format_str = '{:<14.4f} '* 7
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[0,4],C[2,2],C[3,3]))
        elif (LC == 'O'):
            format_str = '{:<14.4f} '* 9
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[1,1],C[1,2],C[2,2],C[3,3],C[4,4],C[5,5]))
        elif (LC == 'M'):
            format_str = '{:<14.4f} '* 13
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,5],C[1,1],C[1,2],C[1,5],C[2,2],C[2,5],C[3,3],C[3,4],C[4,4],C[5,5]))
        elif (LC == 'N'):
            format_str = '{:<14.4f} '* 21
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[0,4],C[0,5],C[1,1],C[1,2],C[1,3],C[1,4],C[1,5],C[2,2],C[2,3],C[2,4],C[2,5],C[3,3],C[3,4],C[3,5],C[4,4],C[4,5],C[5,5]))
        else:
            pass
        format_str = '{:<14.4f} '* 17 + '\n'
        fo.write(format_str.format(E.BV,E.BR,E.BH,E.GV,E.GR,E.GH,E.EV,E.ER,E.EH,E.nuH,E.kH,E.HH,E.AVR,E.AU,E.CL,E.CB,E.Debye))

    def Cinf_model_ref(self, LC='C', fo=None, E=None):
        """
        Writes reference material properties and compliance matrix values to an output file.
        
        The function appends formatted values of the compliance matrix and other material properties
        to the specified output file based on the selected compliance type (LC) for the reference state.
        
        Parameters:
        LC (str): A character representing the type of crystal structure. 
                  Possible values include:
                  - 'C': Cubic
                  - 'H': Hexagonal
                  - 'TI': Tetragonal I
                  - 'TII': Tetragonal II
                  - 'RI': Rhombohedral I
                  - 'RII': Rhombohedral II
                  - 'O': Orthorhombic
                  - 'M': Monoclinic
                  - 'N': Triclinic
        fo (str): The name of the file to which the data will be appended.
        E (object): An object containing material properties and compliance matrix data.

        Returns:
        None: The formatted data is appended to the specified output file.
        """
        
        C = E.C_matrix
        format_str = '{:<14.4f} '* 4
        fo.write(format_str.format(E.T,E.V,E.P,E.rho))

        if (LC == 'C'):
            format_str = '{:<14.4f} '* 3 
            fo.write(format_str.format(C[0,0],C[0,1],C[3,3]))
        elif (LC == 'H'):
            format_str = '{:<14.4f} '* 5
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[2,2],C[3,3]))
        elif (LC == 'TI'):
            format_str = '{:<14.4f} '* 6
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[2,2],C[3,3],C[5,5]))
        elif (LC == 'TII'):
            format_str = '{:<14.4f} '* 7
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,5],C[2,2],C[3,3],C[5,5]))
        elif (LC == 'RI'):
            format_str = '{:<14.4f} '* 6
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[2,2],C[3,3]))
        elif (LC == 'RII'):
            format_str = '{:<14.4f} '* 7
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[0,4],C[2,2],C[3,3]))
        elif (LC == 'O'):
            format_str = '{:<14.4f} '* 9
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[1,1],C[1,2],C[2,2],C[3,3],C[4,4],C[5,5]))
        elif (LC == 'M'):
            format_str = '{:<14.4f} '* 13
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,5],C[1,1],C[1,2],C[1,5],C[2,2],C[2,5],C[3,3],C[3,4],C[4,4],C[5,5]))
        elif (LC == 'N'):
            format_str = '{:<14.4f} '* 21
            fo.write(format_str.format(C[0,0],C[0,1],C[0,2],C[0,3],C[0,4],C[0,5],C[1,1],C[1,2],C[1,3],C[1,4],C[1,5],C[2,2],C[2,3],C[2,4],C[2,5],C[3,3],C[3,4],C[3,5],C[4,4],C[4,5],C[5,5]))
        else:
            pass
        format_str = '{:<14.4f} '* 4 + '\n'
        fo.write(format_str.format(E.BH,E.GH,E.EH,E.Debye))