"""
pySensor: A high throughput FRET analysis software utilizing the sensorFRET 
approach to sensitized emission FRET
"""
from pylab import *
import os
import numpy as np
import pandas as pd
from imreg_dft import transform_img
from scipy.optimize import nnls
from scipy.misc import imresize
from statsmodels import robust
from scipy.stats import f_oneway,mannwhitneyu,ttest_ind,kruskal,linregress 
from skimage.feature import register_translation
from skimage.measure import label
import tifffile as tf
from tqdm import tnrange
import datetime


def unmix(comps,raw):
    """Fast unmixing which allows negative magnitudes (runs NNLS and if any fits are 
    close to zero it will run it again with both positive and negative components)
    """
    c_num=comps.shape[1]
    Mag,res=nnls(comps,raw)
    if any(Mag<.02*np.mean(Mag)):
        coefs,res=nnls(np.hstack([comps,comps*-1]),raw)
        for c in range (c_num):        
            Mag[c]=coefs[c]-coefs[c+c_num]
    return(Mag,res)


class FRET_Calc:

    def __init__(self,name,c_df):
        """Initializes sensorFRET object
            
            X=FRET_Calc(name,calib)
            
            Inputs:
            'name' experiment name, object will be saved as 'name.sf'
            
            'c_df' dataframe containing the calibration parameters for a particular
                    fluorophore and excitation wavelength pairing. This includes 
                    the fluorophore emission shapes, names, extinction coefficient
                    rations, quantum efficiencies, and gamma. See FRET_calib 
                    class for details.
        """   

        #read in calibration data
        
        self.info=c_df.T
        self.calibFromInfo()
        
        now=datetime.datetime.now()
        self.date=now
        self.ExpName=name+'_'+str(now.month)+'-'+str(now.day)+'-'+str(now.year)
        
        
        #dictionary relating short form attribute names to their long form
        self.alias={'I':'Intensity','Mx':'Spectra Maximum',
                    'D':self.fluor_names[0]+' Magnitude',
                    'A':self.fluor_names[1]+' Magnitude',
                    'AF':'Autofluorescence Magnitude',
                    'R':'Normalized Residual',
                    'Alpha':'Alpha: '+self.fluor_names[0]+'('+self.ex_names[1]+' / '+self.ex_names[0]+')',
                    'Beta':'Beta: '+self.fluor_names[1]+'('+self.ex_names[1]+' - Alpha * '+self.ex_names[0]+')',
                    'A_dir':self.fluor_names[1]+' Direct Excitation',
                    'Eff':'FRET Efficiency',
                    'eD_eA':self.fluor_names[0]+' to '+self.fluor_names[1]+' excitation ratio',
                    'Eff_S':'Stoicheometry Corrected FRET Efficiency',
                    'S':'Stoicheometry (['+self.fluor_names[0]+']/['+self.fluor_names[1]+'])',
                    'Ind':'FRET Index',
                    'group':'Group Number','image':'Image Number','region':'Region Number',
                    'x_loc':'X Coordinate','y_loc':'Y Coordinate','area':'Observation Area (Pixels)'}
        
        #dictionary relating long form attribute names to their short form
        #(reverses mapping on self.alias)
        self.alias_inv={v: k for k, v in self.alias.iteritems()}
        
        
        #list of parameters which are excitation independent
        self.ex_independent=['area','group','image','region',
                             'x_loc','y_loc','Alpha','Beta','Eff','Eff_S',
                             'S','group','image','region']
                     
        #colorlist for plotting               
        self.clist=['blue','green','red','purple','orange','magenta','teal',
                    'cyan','olive','darkred','lime','goldenrod','black',
                    'coral','saddlebrown','lightskyblue','steelblue',
                    'grey','peru','palevioletred','tan']


    def calibFromInfo(self):
        self.calib_name=str(self.info['Calibration_Name'][0])
        #Fluorophore Names            
        self.fluor_names=[str(self.info['D_name'][0]),str(self.info['A_name'][0])]
        #Excitation Frequencies Used            
        self.ex_names=[str(self.info['Ex1'][0]),str(self.info['Ex2'][0])]
        #Array containing emission shapes for Donor, Acceptor, and Autofluorescence at ex1
        self.comps1=np.array(eval(self.info['comps1'][0]))
        self.comps2=np.array(eval(self.info['comps2'][0]))
        #emission wavelength vector        
        self.wl_em=np.array(eval(self.info['wl_em'][0]))
        #Number of emission channels
        self.channels=len(self.wl_em)
        #The gamma calibration parameter [(eD2/eD1)*(eA1/eA2)]
        self.gamma=float(self.info['gamma'][0])
        #the extinction coefficient of the donor divided by the acceptor at ex1 and ex2
        self.ex_ratio1=float(self.info['exRatio1'][0])
        self.ex_ratio2=float(self.info['exRatio2'][0])
        #The Quantum Yield of the donor and acceptor fluorophores
        self.Qd=float(self.info['Qd'][0])          
        self.Qa=float(self.info['Qa'][0])

#---------------------------------------------
#
#FUNCTIONS FOR READING AND PRE-PROCESSING IMAGES
#
#---------------------------------------------


        
    def get_nested_subdirectories(self,print_names=False):
        """Recursively reads directories in 'Grouped Tiffs' folder 
           to get all the image filenames for importing
            
            Directories must be structured as such:
            
            Grouped Tiffs
                Experimental Group A
                    Frequency 1
                        F1 tifs (multi channel spectral images)
                    Frequency 2
                        F2 tifs (multi channel spectral images)
                    Masks
                        Mask tifs (single channel binary images)
                Experimental Group B...
                
            then generates the variables self.group_names and self.file_names
            (format: self.group_names[group],dtype=string)
            (format: self.file_names[group][F1,F2,Mask][image],dtype=string)
            
            Keyword Arguments:
            'print_names=True' prints the names of all images for you to check
            
        """
        
        #get group directory list
        root_dir=os.getcwd()+'/Grouped_Tiffs/'        
        g_dir=[]
        for (dirname,dirs,files) in os.walk(root_dir):
            g_dir.append(dirs)
        
        #get file names
        FN=[]
        FN2=[]
        for i in range(len(g_dir[0])):
            t=[]
            t2=[]
            for (dirname,dirs,files) in os.walk(os.path.join(root_dir,g_dir[0][i])):
                flist=[]
                flist2=[]
                for filename in files:
                    if filename.endswith(".tif"):
                        flist.append(os.path.join(dirname,filename))
                        if print_names==True:
                            print(filename)
                        #extracts the image name from filepath
                        flist2.append(filename.split("/")[-1].split(".")[0])
                        
                #removes empty list creation that results from os.walk function
                if len(flist)>0:
                    t.append(flist)
                    t2.append(flist2)
            FN.append(t)
            FN2.append(t2)
        
        #Creates a list of group names based on the folder names in 
        #Grouped_Tiffs to be used for plotting labels
        self.group_names=g_dir[0]

        #Creates a list of image names based on the file names in each of the 
        #group folders to be used for plotting labels
        self.image_names=FN2
        
        #Creates a list of full directory paths for importing image data
        self.file_names=FN
        
        
    def get_tif_container(self,interpolation=False):
        """Imports Tiff data into numpy arrays based on the filenames determined 
            from 'self.get_nested_subdirectories()' and generates the 
            self.tif_array variable
             (format: self.tif_array[group][F1,F2,Mask][image,em,x,y]or[image,x,y](for mask),dtype=float)
            
            Keyword Arguments:
            'interpolation=[Xdim,Ydim]' resizes all images to Xdim by Ydim resolution,
            False uses the original image dimensions but the program will fail 
            if images are not all the same size
            
        """
        
        self.info['interpolation']=[interpolation]
        FN=self.file_names
        #get size of the first image for array containers
        channels_i,x,y=tf.TiffFile(FN[0][0][0]).asarray().shape
        
        if interpolation!=False:
            x=int(interpolation[0])
            y=int(interpolation[1])
        
       # print("channels,x,y",channels_i,x,y)
    
        if self.channels==[]:        
            self.channels=channels_i        
        if channels_i!=self.channels:
            print('Number of Channels in Image Do Not Match Specified Emission Channels. Could be due to extra transmission channel or incorrect specification of the number of emission channels. This will default to the specified number of emission channel')
        self.img_num=[]        
        self.X=x
        self.Y=y
        tif_container=[]
        group_num=[]        
        dim3=[]        
        for group in tnrange(len(FN)):
            #determines number of images in subdir group
            im_count=len(FN[group][0])
            self.img_num.append(im_count)
            #creates ndarray to store all images
            F1=np.zeros((im_count,self.channels,x,y))
            F2=np.zeros((im_count,self.channels,x,y))
            Mlabel=np.ones((im_count,x,y))
            image_num=[]            
            dim2=[]            
            for image in range(im_count):#loops through subdir list and groups all images into array container
                F1t=tf.TiffFile(FN[group][0][image]).asarray()[:self.channels,:,:]
                F2t=tf.TiffFile(FN[group][1][image]).asarray()[:self.channels,:,:]
                xi,yi=F1t.shape[1:]                
                
                for channel in range(self.channels):
                    if interpolation!=False:
                        F1[image,channel,:,:]=imresize(F1t[channel,:,:],size=(x,y),interp='bilinear',mode='F')
                        F2[image,channel,:,:]=imresize(F2t[channel,:,:],size=(x,y),interp='bilinear',mode='F')
                    else:
                        F1[image,channel,:,:]=F1t[channel,:,:]
                        F2[image,channel,:,:]=F2t[channel,:,:]
                
                M1t=np.ones((xi,yi))
                M2t=np.ones((xi,yi))
                
                if len(FN[group])>2:
                    M1t=tf.TiffFile(FN[group][2][image]).asarray()
                    M1t=(M1t/np.max(M1t))
                if len(FN[group])>3:
                    M2t=tf.TiffFile(FN[group][3][image]).asarray()
                    M1t=(M1t/np.max(M1t))

                Mt=M1t*M2t
                if interpolation!=False:
                    Mt=imresize(Mt,size=(x,y),interp='bilinear',mode='F')
                    Mt=np.rint(Mt)
                    
                Mlabel[image,:,:]=label(Mt)
                
    
                #loops through labeled masks and sums up the number of unique 
                #regions in the mask, these get appended to a list since they 
                #are variable in size. 
                region_num=[]
                for r in range(0,int(np.max(Mlabel[image,:,:]))):
                    p=np.sum(Mlabel[image]==(r+1))
                    region_num.append(p)
                image_num.append(region_num)
            group_num.append(image_num)
            #adds all ndimage arrays into a list
            tif_container.append([F1,F2,Mlabel])
    
        #tif_container[exp groups][F1,F2,Mask][img,ch,x,y] or [img,x,y](for Mask)    
        self.shape=group_num
        self.tif_array=tif_container

  
    def register_images(self,to_F1=False):
        """Registers the images from the two excitation frequencies and updates
            self.tif_array variable with the shifted images
            (format: self.tif_array[group][F1,F2,Mask][image,em,x,y]or[image,x,y](for mask),dtype=float)
 
             Keyword Arguments:
             
           'to_F1=bool' option registers frequency 2 image to the frequency 1 image
            as opposed to registering frequency 1 to frequency 2 (default)
        """
        #tif_container[exp groups][F1,F2,Mask][img,ch,x,y] or [img,x,y](for Mask)    
        #registers images to account for image drift (F1 is translated by default)
        tif_container=self.tif_array
        
        self.info['Register to F1']=to_F1
        m_ind=0 #mobile index
        s_ind=1 #stationary index
        if to_F1==True:
            m_ind=1
            s_ind=0
        
        for g in tnrange (len(tif_container)):
            for i in range (tif_container[g][0].shape[0]):
                m_im=np.sum(tif_container[g][m_ind][i,:,:,:],axis=0)
                s_im=np.sum(tif_container[g][s_ind][i,:,:,:],axis=0)
                yshift,xshift=register_translation(s_im,m_im)[0]
                print('Y: '+str(format(yshift,'1.3f')),'X: '+str(format(xshift,'1.3f')))
                lg_shift=(abs(yshift)>(0.05*self.Y))or(abs(xshift)>(0.05*self.X))                 
                
                if lg_shift:
                    check=eval(raw_input('Register using this large registration shift? (True or False)'))
                    if check:
                        for em in range(tif_container[g][0][0,:,:,:].shape[0]):
                            tif_container[g][m_ind][i,em,:,:]=transform_img(tif_container[g][m_ind][i,em,:,:],
                                                                            scale=1.0,angle=0.0,
                                                                            tvec=(yshift,xshift))
                else:
                    for em in range(tif_container[g][0][0,:,:,:].shape[0]):
                        tif_container[g][m_ind][i,em,:,:]=transform_img(tif_container[g][m_ind][i,em,:,:],
                                                                        scale=1.0,angle=0.0,
                                                                        tvec=(yshift,xshift))
            print('_____________')
        self.tif_array=tif_container




#---------------------------------------------
#
#FUNCTIONS FOR ORGANIZING AND ANALYZING DATA
#
#---------------------------------------------

        
    
    def sort_tif_container(self,central_tendency='median'): 
        """Sorts and averages pixels into DataFrames according to observation type 
        (pix_DF,region_DF,image_DF and groupDF)
        
        central_tendency parameter chooses either the mean or the median 
        to find the average spectra. This setting also sets the default variance 
        measurement, standard deviation (SD) for the mean and median average 
        deviation (MAD) for the median.
            
        """
        
        pixels=0
        regions=0
        images=0
        groups=len(self.shape)
        
        #self.shape is a series of nested lists with variable dimensions. The code below parses the total counts of each dimension
        
        # overall container shape, corresponds to number of directories in ~/Calibration_Inputs/
        for g in range (0,len(self.shape)):
            #Number of images in Sub-directory within ~/Calibration_Inputs/sub_dir/, where sub_dir could be [Donor, Acceptor, FRET_Standard,AutoF]
            images=images+len(self.shape[g])
            for i in range (0,len(self.shape[g])):
                #Number of unique regions within a mask generated by the label() function from skimage.measure 
                regions=regions+len(self.shape[g][i])                
                for r in range (0,len(self.shape[g][i])):
                    pixels=pixels+self.shape[g][i][r]

        print('Total Pixels: '+str(pixels))
        print('Total Regions: '+str(regions))
        print('Total Images: '+str(images))
        print('Total Groups: '+str(groups))
        self.counts=[groups,images,regions,pixels]
        
        #sets the central tendency and error functions to be used
        if central_tendency=='median':
            self.ct_name='median'        
            self.ct_function=np.nanmedian
            self.ct_err_name='MAD'        
            self.ct_err_function=robust.mad
        if central_tendency=='mean':
            self.ct_name='mean'        
            self.ct_function=np.nanmean
            self.ct_err_name='SD'        
            self.ct_err_function=np.nanstd
            
        self.info['Central Tendency']=self.ct_name
        self.info['Central Tendency Error']=self.ct_err_name
        
        [groups,images,regions,pixels]=self.counts
        #Groups refers to the number of experimental group sub-directories in 'Grouped_Tiffs' folder
        group_container=np.full([groups,3,self.channels],np.nan)
        
        #Images refer to the number of linked images that make up an image observation: Image1 refers to [Image1_Freq1,Image1_Freq2,Image1_Mask]:
        image_container=np.full([images,3,self.channels],np.nan)
        
        #regions refer to the number of linked regions that make up a region observation: Region1 refers to [Region1_Freq1,Region1_Freq2,Region1_Mask]:
        region_container=np.full([regions,3,self.channels],np.nan)
        
        #pixel refers to number of linked pixels that make up a pixel observation: Pixel1 refers to [Pixel1_Freq1, Pixel1_Freq2,Pixel_Mask]
        pix_container=np.full([pixels,3,self.channels],np.nan)
        pix=0
        reg=0
        img=0
        grp=0
        
        pix_labels=[]        
        reg_labels=[]
        image_labels=[]
        group_labels=[]
        
        for g in tnrange (0,len(self.shape)):
            g_start=pix
            g_name=[self.group_names[g]]
            group_labels.append(g_name[0])
            g_img_count=0
            g_reg_count=0
            for i in range (0,len(self.shape[g])):
                i_start=pix
                i_name=[g_name[0],g_name[0]+': '+self.image_names[g][0][i].replace(self.ex_names[0],'')]
                image_labels.append(i_name)
                i_reg_count=0
                for r in range (1,len(self.shape[g][i])+1):
                    r_start=pix
                    r_name=[g_name[0],i_name[1],i_name[1]+': R-'+str(r)]
                    reg_labels.append(r_name)
                
                    mask_coor = np.nonzero(self.tif_array[g][2][i,:,:]==r)
                    for (x,y) in zip(*mask_coor):
                        p_name=[g_name[0],i_name[1],r_name[2],r_name[2]+': ('+str(x)+','+str(y)+')']
                        pix_labels.append(p_name)
                        pix_container[pix,0,:self.channels]=self.tif_array[g][0][i,:,x,y]
                        pix_container[pix,1,:self.channels]=self.tif_array[g][1][i,:,x,y]
                        pix_container[pix,2,0:6]=g,i,r,x,y,1
                        pix=pix+1
                       
                    region_container[reg,:,:self.channels]=self.ct_function(pix_container[r_start:pix,:,:self.channels],axis=0)
                    region_container[reg,:,:self.channels]=self.ct_function(pix_container[r_start:pix,:,:self.channels],axis=0)
                    region_container[reg,2,5]=(pix-r_start)
                    reg=reg+1
                    g_reg_count=g_reg_count+1
                    i_reg_count=i_reg_count+1
                image_container[img,:,:self.channels]=self.ct_function(pix_container[i_start:pix,:,:self.channels],axis=0)
                image_container[img,:,:self.channels]=self.ct_function(pix_container[i_start:pix,:,:self.channels],axis=0)
                image_container[img,2,5]=(pix-i_start)
                image_container[img,2,2]=i_reg_count               
                img=img+1
                g_img_count=g_img_count+1
            group_container[grp,:,:self.channels]=self.ct_function(pix_container[g_start:pix,:,:self.channels],axis=0)
            group_container[grp,:,:self.channels]=self.ct_function(pix_container[g_start:pix,:,:self.channels],axis=0)
            group_container[grp,2,5]=(pix-g_start)
            group_container[grp,2,2]=g_reg_count
            group_container[grp,2,1]=g_img_count
            grp=grp+1
        #subroutine to convert array to a labeled Dataframe
        def container_to_DF(container,labels):
            
            dat={'spectra1':container[:,0,:].tolist(),
                 'spectra2':container[:,1,:].tolist(),
                 'group':container[:,2,0],
                 'image':container[:,2,1],
                 'region':container[:,2,2],
                 'y_loc':container[:,2,3],
                 'x_loc':container[:,2,4],
                 'area':container[:,2,5]}
            DF=pd.DataFrame(data=dat)                        
            DF.insert(0,'label',labels)
            return(DF)
            
        self.pix_DF=container_to_DF(pix_container,pix_labels)
        self.region_DF=container_to_DF(region_container,reg_labels)
        self.image_DF=container_to_DF(image_container,image_labels)
        self.group_DF=container_to_DF(group_container,group_labels)


    def unmix_DF(self,observation='all'):
        """Cycles through a DataFrame and unmixes the spectra into different 
        fitting components that are then saved into the DataFrame
        
        observation='all' option unmixes all dataframes or a list of observations
        types can be specified, for example: observation=["group","image",region]        
        """
        if observation=='all':
            obs=['group','image','region','pix']
        else:
            obs=observation
        
        self.DF_names=[]
        
        for o in obs:
            self.DF_names.append(o+"_DF")
            DF=getattr(self,o+'_DF')
            spectra1=DF.spectra1.values
            spectra2=DF.spectra2.values
            params=[]      
            for i in tnrange(len(DF.index)):
                params.append(self.unmix_magnitudes(spectra1[i],spectra2[i],self.comps1,self.comps2))
            params=np.array(params).T
            #params_list must coorespond to the order of returned parameters from unmix_magnitudes
            param_list=['D1','A1','AF1','R1','I1','Mx1',
                        'D2','A2','AF2','R2','I2','Mx2']
            
            for p in range(len(param_list)):        
                DF[param_list[p]]=params[p]
                
                
    def unmix_magnitudes(self,spectra1,spectra2,comps1,comps2):
        """Takes paired spectra and unmixing matrices from excitations 1 and 2 
        as input. Returns an object containing all magnitudes needed for any 
        processing"""
        
        #Compute Area under spectra, integrated intensity
        I1=np.trapz(spectra1)
        I2=np.trapz(spectra2)
        Mx1=np.max(spectra1)
        Mx2=np.max(spectra2)
        
        #Unmix Donor,Acceptor,AF signal Magnitudes for FRET spectra, denoted with F_
        [D1,A1,AF1],R1=unmix(comps1.T,spectra1)
        [D2,A2,AF2],R2=unmix(comps2.T,spectra2)
        
        return([D1,A1,AF1,R1/I1,I1,Mx1,
                D2,A2,AF2,R2/I2,I2,Mx2])
        
    def new_calculation(self,calc,name=False):
        """Allows new parameters to be created based off of simple operations,
        for example the normalized donor magnitude can be generated by:
        
        self.new_calculation(['D1','/','I1'],name='Dnorm_1')
        
        
        """
        operators= {'+': (lambda x,y: x+y),
                    '-': (lambda x,y: x-y),
                    '*': (lambda x,y: x*y),
                    '/': (lambda x,y: x/y),
                    '^': (lambda x,y: x**y)}
        if name==False:
            name=calc[0]+calc[1]+calc[2]
        
        try:
            self.pix_DF[name]=operators[calc[1]](
                             getattr(self.pix_DF,calc[0]),
                             getattr(self.pix_DF,calc[2]))
        except:
            print('Pix Calculation Failed')
        try:
            self.region_DF[name]=operators[calc[1]](
                             getattr(self.region_DF,calc[0]),
                             getattr(self.region_DF,calc[2]))
        except:
            print('Region Calculation Failed') 
        try:
            self.image_DF[name]=operators[calc[1]](
                             getattr(self.image_DF,calc[0]),
                             getattr(self.image_DF,calc[2]))
        except:
            print('Image Calculation Failed')
        try:
            self.group_DF[name]=operators[calc[1]](
                             getattr(self.group_DF,calc[0]),
                             getattr(self.group_DF,calc[2]))
        except:
            print('Group Calculation Failed') 
        
        #update parameter dictionaries & lists
        self.alias[name]=name
        self.alias_inv={v: k for k, v in self.alias.iteritems()}
        self.ex_independent.append(name)
        
    def default_calculations(self):
        """Generates the standard parameters needed for the FRET analysis
            ie. alpha,beta,direct excitation, and efficiency
        """
        
        for df in self.DF_names:
            DF=getattr(self,df)
            
            DF['Ind1']=DF.A1/DF.D1
            DF['Ind2']=DF.A2/DF.D2
            
            DF['Alpha']=DF.D2/DF.D1
            DF['Beta']=DF.A2-DF.Alpha*DF.A1
            
            DF['A_dir1']=DF.Beta/(DF.Alpha*(self.gamma**-1-1))
            DF['A_dir2']=DF.Beta/(1-self.gamma)
            
            DF['Eff']=(((DF.D2*self.Qa)/((DF.A2-DF.A_dir2)*self.Qd))+1)**(-1)
            
            DF['eD_eA1']=(DF.D1/self.Qd+(DF.A1-DF.A_dir1)/self.Qa)/(DF.A_dir1/self.Qa)
            DF['eD_eA2']=(DF.D2/self.Qd+(DF.A2-DF.A_dir2)/self.Qa)/(DF.A_dir2/self.Qa)
            
            DF['S']=DF.eD_eA2*self.ex_ratio2**-1
        
            DF['Eff_S']=(DF.A2-DF.A_dir2)/DF.A_dir2*self.ex_ratio2**-1



#---------------------------------------------
#
#SUBROUTINES FOR PLOTTING FUNCTIONS
#
#---------------------------------------------



    def specify_mask(self,DF,s):
        """Excludes data from DataFrame unless it meet the conditions fed in 
        from s.
        
        s format: 
        ['parameter','logic operator', [list of values(can be length 1)]]
        """
        
        logic={"<": (lambda x,y: x<y[0]), 
               ">": (lambda x,y: x>y[0]),
               "><": (lambda x,y: (x>y[0])&(x<y[1])),
               "=": (lambda x,y: x.isin(y)),
               "!=": (lambda x,y: -x.isin(y))}
        #    chosen function (DF.parameter,value list)
        mask=logic[s[1]] (getattr(DF,s[0]),s[2])
            
        return(DF[mask])
    
    def get_figure_sets(self, DF, param, specify, plot_by='group'):
        """ Gets the data and labels for a plotting function from a observation 
        dataframe based on the parameter requested, the type of plot, and any 
        conditional statements in specify
        """
       
       
        if plot_by=='group':
            cutoff=0
        if plot_by=='image':
            cutoff=1
        if plot_by=='region':
            cutoff=2
            
        
        DF_masked=DF.copy()
        for s in specify:
            DF_masked=self.specify_mask(DF_masked,s)
        
        dat=np.array(getattr(DF_masked,param).values)
        labels=np.vstack(DF_masked.label.values)
        temp_labels=[]
        for obs in range(len(labels)):
            #appends location id up to the cutoff determined by plot_by
            temp_labels.append(str(labels[obs][cutoff]))
        temp_labels=np.array(temp_labels)
        fig_labels=np.unique(temp_labels)
        
        Data=[]
        for n in fig_labels:
            Data.append(dat[temp_labels==n])
        return(Data,fig_labels)   
        



#---------------------------------------------
#
#PLOTTING FUNCTIONS
#
#---------------------------------------------
        
    def spectraVariance(self,ex=1,observation='region',plot_by='group',
                        specify=[],colors=[]):
        """Plots the noise characteristics of the spectra. First panel shows the
        average spectra with a filled band 
        """
        def genSpectraplot(self,Spectra_array,sub1,sub2,sub3,sub4,l,c):
            Spectra=Spectra_array.T/np.trapz(Spectra_array,axis=1)
            Spectra_ave=np.nanmean(Spectra,axis=1)
            Spectra_stdev=np.nanstd(Spectra,axis=1)
            N=Spectra.shape[1]
            sub1.plot(self.wl_em,Spectra_ave,color=c,ls='--',label=l+' (n='+str(N)+')')
            sub1.fill_between(self.wl_em,(Spectra_ave+Spectra_stdev),(Spectra_ave-Spectra_stdev),color=c,alpha=.25)
            sub1.set_xlabel('Wavelength')
            sub1.set_ylabel('Normalized Intensity')
            
            sub3.plot(self.wl_em,Spectra_stdev/(Spectra_ave**.5),ls='',marker='o',color=c)
            sub3.set_xlabel('Wavelength')
            sub3.set_ylabel('Slope')
            
            sub2.plot((np.mean(Spectra_array,axis=0)**.5),Spectra_stdev,ls='',marker='o',color=c)
            sub2.set_ylabel('Standard Deviation')
            sub2.set_xlabel('Intensity^.5')
            
            sub4.axis('off')            
            
            sub1.legend(loc='upper left', bbox_to_anchor=(1.18,-.18))        
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        param_ex='spectra'+str(ex)
        add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)
              
        Varfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=[15,16])
        
        for i in range(len(data)):
            genSpectraplot(self,np.vstack(data[i]),sub1,sub2,sub3,sub4,
                          labels[i]+add_label,colors[i%len(colors)]) 
        
    
    def showImage(self,param='Eff',ex=2,mn=0,mx=1,specify=[],cmap='jet'):
        
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=self.pix_DF.copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by='image')
        xloc,labels=self.get_figure_sets(DF,'x_loc',specify,plot_by='image')
        yloc,labels=self.get_figure_sets(DF,'y_loc',specify,plot_by='image')
        
        for i in range(len(data)):
            figure()
            image=np.full([self.X,self.Y],np.nan)
            imshow(np.ones([self.X,self.Y]),cmap='Greys_r')
            for p in range(len(data[i])):
                image[int(yloc[i][p]),int(xloc[i][p])]=data[i][p]
            imshow(image,vmin=mn,vmax=mx,cmap=cmap),colorbar(label=self.alias[param]+add_label)
            title(labels[i]+add_label)
            axis('off')
            
            
    def plotHistogram(self,param='D',ex=2,lim_x=[0,1],bins=100,norm=False,
                      observation='region',plot_by='group',specify=[],colors=[],print_ct=True):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)

        for i in range(len(data)):
            ct=self.ct_function(data[i])
            ct_err=self.ct_err_function(data[i])
            if print_ct==True:
                print(labels[i])
                print(self.ct_name+': '+str(ct))
                print(self.ct_err_name+': '+str(ct_err))
            axvline(ct, color=colors[i%len(colors)],
                    ls='--',label=labels[i]+' ('+self.ct_name+')')
            hist(data[i],histtype='stepfilled',alpha=.5,color=colors[i%len(colors)],
                 bins=bins,range=(lim_x[0],lim_x[1]),normed=norm) 
            
        legend()
        xlabel(self.alias[param]+add_label)
        ylabel('Frequency')
        
    def plotScatter(self,param_x='D',ex_x=2,param_y='D',ex_y=2,
                    lim_x=[0,1],lim_y=[0,1],observation='region',plot_by='group',
                    specify=[],colors=[],alpha=1,fit=False,print_fit=False,ls=''):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param_x in self.alias_inv:
            param_x=self.alias_inv[param_x]
        
        if param_y in self.alias_inv:
            param_y=self.alias_inv[param_y]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param_x in self.ex_independent:
            param_x_ex=param_x            
            add_label_x=''
        else:
            param_x_ex=param_x+str(ex_x)
            add_label_x=' ('+self.ex_names[ex_x-1]+')'
        
        if param_y in self.ex_independent:
            param_y_ex=param_y            
            add_label_y=''
        else:
            param_y_ex=param_y+str(ex_y)
            add_label_y=' ('+self.ex_names[ex_y-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data_x,labels=self.get_figure_sets(DF,param_x_ex,specify,plot_by=plot_by)
        data_y,labels=self.get_figure_sets(DF,param_y_ex,specify,plot_by=plot_by)
        
        for i in range(len(data_x)):
            plot(data_x[i],data_y[i],ls=ls,marker='o',alpha=alpha,
                 color=colors[i%len(colors)],label=labels[i]) 
            if fit==True:
                f=linregress(data_x[i],data_y[i])
                x_fit=np.linspace(np.nanmin(data_x[i]),np.nanmax(data_x[i]),100)
                y_fit=f[0]*(x_fit)+f[1]
                plot(x_fit,y_fit,color=colors[i%len(colors)],ls='--',label='Fit')
                if print_fit==True:
                    print(labels[i])
                    print('Fit Slope='+str(format(f[0],'1.3f')))
                    print('Fit Intercept='+str(format(f[1],'1.3f')))
                    print('Fit R^2='+str(format(f[2]**2,'1.3f')))
                    print('Fit P-Value='+str(format(f[3],'1.3f')))
        legend()
        xlabel(self.alias[param_x]+add_label_x)
        ylabel(self.alias[param_y]+add_label_y)
        xlim(lim_x[0],lim_x[1])
        ylim(lim_y[0],lim_y[1])
        
    def plotFRET(self,observation='region',plot_by='group', specify=[],print_params=False):
        
        def genFRETplot(self,F1,F2,name,print_params=False):
            """Generates the standard four figure plot that shows the calculation
            steps. Used within self.calcE_by(group/image/region)
            
            """    
            
            [D1,A1,AF1]=unmix(self.comps1.T,F1)[0]
            [D2,A2,AF2]=unmix(self.comps2.T,F2)[0]
    
            alpha=D2/D1
            beta=A2-alpha*A1
            A1_dir=beta/(alpha*(self.gamma**-1-1))
            A2_dir=beta/(1-self.gamma)
            Eff=(((D1*self.Qa)/((A1-A1_dir)*self.Qd))+1)**(-1)
            
            
            #Eff2=(((D2*self.Qa)/((A2-A2_dir)*self.Qd))+1)**(-1)
            
            FRETfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=(17,17))
            FRETfig.suptitle(name+' (E= '+'{0:.3f}'.format(Eff)+')',fontsize=30)
            
            sub1.set_title('Alpha Fit')
            sub1.plot(self.wl_em,F1,label='$F_{DA}^{1}$ Raw',ls='-',color='red')
            sub1.plot(self.wl_em,F2,label='$F_{DA}^{2}$ Raw',ls='--',color='red')
            sub1.plot(self.wl_em,D1*self.comps1[0,:],label='$F_{DA}^{1}$ Donor Fit',ls='-',color='blue',lw=3)
            sub1.plot(self.wl_em,D2*self.comps2[0,:],label='$F_{DA}^{2}$ Donor Fit',ls='--',color='blue',lw=3)
            sub1.plot(self.wl_em,AF1*self.comps1[2,:],label='$F_{DA}^{1}$ AF Fit',ls='-',color='green',lw=3)
            sub1.plot(self.wl_em,AF2*self.comps2[2,:],label='$F_{DA}^{2}$ AF Fit',ls='--',color='green',lw=3)
            sub1.set_xlabel('Wavelength')
            sub1.set_ylabel('Intensity')
            sub1.legend(loc='best')
            sub1.set_xlim([self.wl_em[0],self.wl_em[-1]])
            
            
            sub2.set_title('Beta Fit')
            sub2.plot(self.wl_em,((F2-AF2*self.comps2[2,:])-alpha*(F1-AF1*self.comps1[2,:])),c="blue",label='$F_{DA}^{2}- Alpha * F_{DA}^{1}$')
            sub2.plot(self.wl_em,self.comps2[1]*beta,c="red",ls='--',label='$\hat{e}_{A}$ * Beta',lw=4)
            sub2.set_xlabel('Emission Wavelength (nm)')
            sub2.set_ylabel('Intensity (au)')
            sub2.legend(loc='best')
            sub2.set_xlim([self.wl_em[0],self.wl_em[-1]])
            
            F1cor=F1-A1_dir*self.comps1[1]
            F2cor=F2-A2_dir*self.comps2[1]
        
            sub3.set_title('Subtraction of Direct Excitation')
            sub3.plot(self.wl_em,F1,ls='-',color='red',lw=2,label='$F_{DA}^{1}$ Raw')
            sub3.plot(self.wl_em,F2,ls='--',color='red',lw=2,label='$F_{DA}^{2}$ Raw')
            sub3.plot(self.wl_em,F1cor,ls='-',color='teal',lw=3,label='$F_{DA}^{1}$ Corrected')
            sub3.plot(self.wl_em,F2cor,ls='--',color='teal',lw=3,label='$F_{DA}^{2}$ Corrected')
            sub3.set_xlabel('Wavelength')
            sub3.set_ylabel('Intensity (au)')
            sub3.set_xlim([self.wl_em[0],self.wl_em[-1]])
            sub3.legend(loc='best',fontsize=12)
        
            sub4.set_title('Unmixed Components')
            sub4.plot(self.wl_em,F1cor-AF1*self.comps1[2,:],color='teal',ls='-',label='$F_{DA}^{1}$ Corrected',lw=2)
            sub4.plot(self.wl_em,self.comps1[0]*D1,color='blue',ls='-',label='$F_{DA}^{1}$ Donor Fit',lw=3)
            sub4.plot(self.wl_em,self.comps1[1]*(A1-A1_dir),color='orange',ls='-',label='$F_{DA}^{1}$ Acceptor Fit',lw=3)
            sub4.plot(self.wl_em,F2cor-AF1*self.comps2[2,:],color='teal',ls='--',label='$F_{DA}^{2}$ Corrected',lw=2)
            sub4.plot(self.wl_em,self.comps2[0]*D2,color='blue',ls='--',label='$F_{DA}^{2}$ Donor Fit',lw=3)
            sub4.plot(self.wl_em,self.comps2[1]*(A2-A2_dir),color='orange',ls='--',label='$F_{DA}^{2}$ Acceptor Fit',lw=3)
            sub4.set_xlabel('Wavelength')
            sub4.set_ylabel('Intensity (au)')
            sub4.set_xlim([self.wl_em[0],self.wl_em[-1]])
            sub4.legend(loc='best',fontsize=12)
            
        if observation=="pixel":
            observation="pix"
            
        DF=getattr(self,(observation+'_DF')).copy()
        
        spec1,labels=self.get_figure_sets(DF,'spectra1',specify,plot_by=plot_by)
        spec2,labels=self.get_figure_sets(DF,'spectra2',specify,plot_by=plot_by)
        
        
        for i in range(len(spec1)):
            ave_spec1=self.ct_function(np.vstack(spec1[i]),axis=0)
            ave_spec2=self.ct_function(np.vstack(spec2[i]),axis=0)
            genFRETplot(self,ave_spec1,ave_spec2,labels[i],print_params)
            
    def spectraFit(self,norm=True,observation='group',plot_by='group',specify=[],colors=[]):
         
        def plotter(self,F1,F2,D1,D2,A1,A2,AF1,AF2,l):
            #central tendency, could be mean or median
            F_ct1=self.ct_function(F1,axis=0)
            F_ct2=self.ct_function(F2,axis=0)
            F_var1=self.ct_err_function(F1,axis=0)
            F_var2=self.ct_err_function(F2,axis=0)
                   
            D1_ct=self.ct_function(D1,axis=0)
            D2_ct=self.ct_function(D2,axis=0)            
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            
            A1_ct=self.ct_function(A1,axis=0)
            A2_ct=self.ct_function(A2,axis=0)            
            A_var1=self.ct_err_function(A1,axis=0)
            A_var2=self.ct_err_function(A2,axis=0)
            
            AF1_ct=self.ct_function(AF1,axis=0)
            AF2_ct=self.ct_function(AF2,axis=0)            
            AF_var1=self.ct_err_function(AF1,axis=0)
            AF_var2=self.ct_err_function(AF2,axis=0)
            
            N=F1.shape[0]
            Varfig, (ax1,ax2) =subplots(1,2,figsize=[15,8])
            Varfig.suptitle(l +' (n='+str(N)+')',fontsize=25)
            
            ax1.plot(self.wl_em,F_ct1,color="red",ls='-',label= self.ct_name+" FRET "+self.ex_names[0])
            ax2.plot(self.wl_em,F_ct2,color="red",ls='-',label=self.ct_name+" FRET "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(F_ct1+F_var1),(F_ct1-F_var1),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(F_ct2+F_var2),(F_ct2-F_var2),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,D1_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,D2_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D1_ct+D_var1),(D1_ct-D_var1),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(D2_ct+D_var2),(D2_ct-D_var2),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,A1_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,A2_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(A1_ct+A_var1),(A1_ct-A_var1),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(A2_ct+A_var2),(A2_ct-A_var2),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,AF1_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[0])
            ax2.plot(self.wl_em,AF2_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(AF1_ct+AF_var1),(AF1_ct-AF_var1),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(AF2_ct+AF_var2),(AF2_ct-AF_var2),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            
            ax1.set_title(self.ex_names[0]+' nm')
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best',fontsize=12)
            
            ax2.set_title(self.ex_names[1]+' nm')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Normalized Intensity [Au]')
            ax2.legend(loc='best',fontsize=12)
        
        #get observation Dataframe
        if observation=="pixel":
            observation="pix"   
        DF=getattr(self,(observation+'_DF')).copy()
        
        #Generates a plot for each plot_by feature
        for i in tnrange(1):
            print("Getting Figure Sets")
            print("---------------------------------------------------------------------------------")
            #get filtered data (plot_by +specify filtering)
            spec1,labels=self.get_figure_sets(DF,"spectra1",specify,plot_by=plot_by)
            D1dat,labels=self.get_figure_sets(DF,"D1",specify,plot_by=plot_by)
            A1dat,labels=self.get_figure_sets(DF,"A1",specify,plot_by=plot_by)
            AF1dat,labels=self.get_figure_sets(DF,"AF1",specify,plot_by=plot_by)
            
            
            spec2,labels=self.get_figure_sets(DF,"spectra2",specify,plot_by=plot_by)
            D2dat,labels=self.get_figure_sets(DF,"D2",specify,plot_by=plot_by)
            A2dat,labels=self.get_figure_sets(DF,"A2",specify,plot_by=plot_by)
            AF2dat,labels=self.get_figure_sets(DF,"AF2",specify,plot_by=plot_by)
            
        print("Generating Plots")
        print("---------------------------------------------------------------------------------")
        for i in tnrange(len(spec1)):
            
            #unpack variables for computation
            D1,D2=D1dat[i],D2dat[i]
            A1,A2=A1dat[i],A2dat[i]
            AF1,AF2=AF1dat[i],AF2dat[i]
            F1=np.vstack(spec1[i])
            F2=np.vstack(spec2[i])
            
            comp_D1=np.tile(self.comps1[0],(len(D1),1))#converts comps array to shape of D1 for simple element wise matrix multiplication
            comp_D2=np.tile(self.comps2[0],(len(D2),1))#converts comps array to shape of D2 for simple element wise matrix multiplication
            comp_A1=np.tile(self.comps1[1],(len(A1),1))#converts comps array to shape of A1 for simple element wise matrix multiplication
            comp_A2=np.tile(self.comps1[1],(len(A2),1))#converts comps array to shape of A2 for simple element wise matrix multiplication
            comp_AF1=np.tile(self.comps1[2],(len(AF1),1))#converts comps array to shape of F1 for simple element wise matrix multiplication
            comp_AF2=np.tile(self.comps2[2],(len(AF2),1))#converts comps array to shape of F2 for simple element wise matrix multiplication
            
            #perform computations
            D1comp=(D1*comp_D1.T).T #Donor1 vectors
            D2comp=(D2*comp_D2.T).T #Donor2 vectors
            A1comp=(A1*comp_A1.T).T #Acceptor1 vectors
            A2comp=(A2*comp_A2.T).T #Acceptor2 vectors
            AF1comp=(AF1*comp_AF1.T).T #AutoF1 vectors
            AF2comp=(AF2*comp_AF2.T).T #AutoF2 vectors
            
            Mag1=np.trapz(F1,axis=1)
            Mag2=np.trapz(F2,axis=1)
            
            if norm==False:
                Mag1=np.ones(Mag1.shape)
                Mag2=np.ones(Mag2.shape)
            
            F1norm=(F1.T/Mag1).T
            F2norm=(F2.T/Mag2).T
            D1norm=(D1comp.T/Mag1).T
            D2norm=(D2comp.T/Mag2).T
            A1norm=(A1comp.T/Mag1).T
            A2norm=(A2comp.T/Mag2).T
            AF1norm=(AF1comp.T/Mag1).T
            AF2norm=(AF2comp.T/Mag2).T
            
            #call plotting functions 
            plotter(self,F1norm,F2norm,D1norm,D2norm,A1norm,A2norm,AF1norm,AF2norm,labels[i])
                
      
      
    def plotFRETvariance(self,observation='region',plot_by='group', specify=[],autoF=True):

        def alpha_plotter(self,ax1,F1,F2,D1,D2,label,color):
            #central tendency, could be mean or median
            F_ct1=self.ct_function(F1,axis=0)
            F_ct2=self.ct_function(F2,axis=0)
            F_var1=self.ct_err_function(F1,axis=0)
            F_var2=self.ct_err_function(F2,axis=0)
                   
            D1_ct=self.ct_function(D1,axis=0)
            D2_ct=self.ct_function(D2,axis=0)            
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            
            ax1.plot(self.wl_em,F_ct1,color="red",ls='-',label= self.ct_name+" FRET "+self.ex_names[0])
            ax1.plot(self.wl_em,F_ct2,color="red",ls='--',label=self.ct_name+" FRET "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(F_ct1+F_var1),(F_ct1-F_var1),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(F_ct2+F_var2),(F_ct2-F_var2),color="red",alpha=.25)
            
            ax1.plot(self.wl_em,D1_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,D2_ct,color="blue",ls='--',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D1_ct+D_var1),(D1_ct-D_var1),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(D2_ct+D_var2),(D2_ct-D_var2),color="blue",alpha=.25)
            
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best',fontsize=12)
            ax1.set_title("Raw Spectra and Alpha")
            
            
        def Beta_plotter(self,ax1,beta_spectra,name):
            #normalize the spectra
            Spectra_ct=self.ct_function(np.nan_to_num(beta_spectra),axis=0)
            Spectra_var=self.ct_err_function(np.nan_to_num(beta_spectra),axis=0)
            
            #peak fitted acceptor component
            acceptor_ref=np.trapz(Spectra_ct)*(self.comps1[1]/np.trapz(self.comps1[1]))
            
            ax1.plot(self.wl_em,Spectra_ct,color="green",ls='-',label=self.ct_name+' Beta')
            ax1.plot(self.wl_em,acceptor_ref,color="red",label=self.fluor_names[1]+" component",ls="--")
            ax1.fill_between(self.wl_em,(Spectra_ct+Spectra_var),(Spectra_ct-Spectra_var),color="green",alpha=.25,
                              label=self.ct_err_name+" $\pm$"+"1")
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.set_title("Beta")
            ax1.legend()
            
        def Spectra_corr_plotter(self,ax1,F1cor,F2cor,name):
            Spectra_ct1=self.ct_function(F1cor,axis=0)
            Spectra_ct2=self.ct_function(F2cor,axis=0)
    
            Spectra_var1=self.ct_err_function(F1cor,axis=0)
            Spectra_var2=self.ct_err_function(F2cor,axis=0)
    
            ax1.plot(self.wl_em,Spectra_ct1,color="purple",ls='-',label='Corrected FRET '+self.ex_names[0])
            ax1.plot(self.wl_em,Spectra_ct2,color="purple",ls='--',label='Corrected FRET '+self.ex_names[1])
            
            ax1.fill_between(self.wl_em,(Spectra_ct1+Spectra_var1),(Spectra_ct1-Spectra_var1),color="purple",alpha=.25,
                                  label="$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Spectra_ct2+Spectra_var2),(Spectra_ct2-Spectra_var2),color="purple",alpha=.25,
                                  label="$\pm$"+"1"+self.ct_err_name)
                
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.set_title("Corrected Spectra")
            ax1.legend()
           
            
        def unmix_corr_plotter(self,ax1,D1,D2,A1cor,A2cor,label):
    
            
            #central tendency, could be mean or median
            D_ct1=self.ct_function(D1,axis=0)
            D_ct2=self.ct_function(D2,axis=0)
            Acor_ct1=self.ct_function(A1cor,axis=0)
            Acor_ct2=self.ct_function(A2cor,axis=0)
        
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            Acor_var1=self.ct_err_function(A1cor,axis=0)
            Acor_var2=self.ct_err_function(A2cor,axis=0)

            ax1.plot(self.wl_em,D_ct1,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,D_ct2,color="blue",ls='--',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.plot(self.wl_em,Acor_ct1,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,Acor_ct2,color="orange",ls='--',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D_ct1+D_var1),(D_ct1-D_var1),color="blue",alpha=.25,
                                  label=self.fluor_names[0]+" $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(D_ct2+D_var2),(D_ct2-D_var2),color="blue",alpha=.25)
            
            ax1.fill_between(self.wl_em,(Acor_ct1+Acor_var1),(Acor_ct1-Acor_var1),color="orange",alpha=.25,
                                  label=self.fluor_names[1]+" $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Acor_ct2+Acor_var2),(Acor_ct2-Acor_var2),color="orange",alpha=.25)
            ax1.set_title("FRET Components")
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best')
    
        #End of subroutines
                
        colors=self.clist
        
        if observation=="pixel":
            observation="pix"
        #get observation level dataframe    
        DF=getattr(self,(observation+'_DF')).copy()
        
        for i in tnrange(1):
            print("Getting Figure Sets")
            print("---------------------------------------------------------------------------------")
            #get filtered data (plot_by +specify filtering)
            spec1,labels=self.get_figure_sets(DF,"spectra1",specify,plot_by=plot_by)
            D1dat,labels=self.get_figure_sets(DF,"D1",specify,plot_by=plot_by)
            A1dat,labels=self.get_figure_sets(DF,"A1",specify,plot_by=plot_by)
            F_AF1,labels=self.get_figure_sets(DF,"AF1",specify,plot_by=plot_by)
            A_dir1,labels=self.get_figure_sets(DF,"A_dir1",specify,plot_by=plot_by)
            
            
            spec2,labels=self.get_figure_sets(DF,"spectra2",specify,plot_by=plot_by)
            D2dat,labels=self.get_figure_sets(DF,"D2",specify,plot_by=plot_by)
            A2dat,labels=self.get_figure_sets(DF,"A2",specify,plot_by=plot_by)
            F_AF2,labels=self.get_figure_sets(DF,"AF2",specify,plot_by=plot_by)
            A_dir2,labels=self.get_figure_sets(DF,"A_dir2",specify,plot_by=plot_by)
            
            
            alpha,labels=self.get_figure_sets(DF,"Alpha",specify,plot_by=plot_by)
            
            
        print("Generating Plots")
        print("---------------------------------------------------------------------------------")
        for i in tnrange(len(spec1)):
            N=spec1[i].shape[0]
            Varfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=[15,16])
            Varfig.suptitle(labels[i]+' (n='+str(N)+')',fontsize=25)
            
            #unpack variables for computation
            AF1,AF2=F_AF1[i],F_AF2[i]
            Alpha=alpha[i]
            F1=np.vstack(spec1[i])
            F2=np.vstack(spec2[i])
            D1=D1dat[i]
            D2=D2dat[i]
            
            comp_D1=np.tile(self.comps1[0],(len(D1),1))#converts comps array to shape of D1 for simple element wise matrix multiplication
            comp_D2=np.tile(self.comps2[0],(len(D2),1))#converts comps array to shape of D2 for simple element wise matrix multiplication
            comp_A1=np.tile(self.comps1[1],(len(D1),1))#converts comps array to shape of A1 for simple element wise matrix multiplication
            comp_A2=np.tile(self.comps1[1],(len(D1),1))#converts comps array to shape of A2 for simple element wise matrix multiplication
            comp_AF1=np.tile(self.comps1[2],(len(F1),1))#converts comps array to shape of F1 for simple element wise matrix multiplication
            comp_AF2=np.tile(self.comps2[2],(len(F2),1))#converts comps array to shape of F2 for simple element wise matrix multiplication
            
            #perform computations
            D1comp=(D1*comp_D1.T).T#Donor1 vectors
            D2comp=(D2*comp_D2.T).T#Donor2 vectors
            A1_dex=(A_dir1[i]*comp_A1.T).T#A_dir1 vectors
            A2_dex=(A_dir2[i]*comp_A2.T).T#A_dir2 vectors
            A1cor=A1dat[i]-A_dir1[i]#corrected acceptor magnitudes
            A2cor=A2dat[i]-A_dir2[i]#corrected acceptor magnitudes
            A1cor=(A1cor*comp_A1.T).T#corrected acceptor vectors 
            A2cor=(A2cor*comp_A2.T).T#corrected acceptor vectors
            AF1comp=(AF1*comp_AF1.T).T#Autofluorescent 1 vectors
            AF2comp=(AF2*comp_AF2.T).T#Autofluorescent 2 vectors
            F1sub=((F1-AF1comp).T*Alpha).T#Subtract autofluor and correct by alpha[vectors]
            F2sub=(F2-AF2comp)#subtract autofluor[vectors]
            beta=F2sub-F1sub#compute beta vectors
            F1cor=(F1-AF1comp)-A1_dex#subtract direct excitation[vectors]
            F2cor=F2sub-A2_dex#subtract direct excitation[vectors]
            
            Mag1=np.trapz(F1,axis=1)
            Mag2=np.trapz(F2,axis=1)
            F1norm=(F1.T/Mag1).T
            F2norm=(F2.T/Mag2).T
            D1norm=(D1comp.T/Mag1).T
            D2norm=(D2comp.T/Mag2).T
            AF1norm=(AF1comp.T/Mag1).T
            AF2norm=(AF2comp.T/Mag2).T
            alpha_norm=D2norm/D1norm
            beta_norm=(F2norm-AF2norm)-(F1norm-AF1norm)*alpha_norm
            F1cor_norm=(F1cor.T/Mag1).T#normalize spectra by magnitude[vectors]
            F2cor_norm=(F2cor.T/Mag2).T#normalize spectra by magnitude[vectors]
            A1cor_norm=(A1cor.T/Mag1).T
            A2cor_norm=(A2cor.T/Mag2).T
            
            #call plotting functions 
            alpha_plotter(self,sub1,F1norm-AF1norm,F2norm-AF2norm,D1norm,D2norm,labels[i],colors[i%len(colors)])#plot raw spectra variance and donor fit variance
            Beta_plotter(self,sub2,beta_norm,labels[i])#plot beta curve variance: excitation independent
            Spectra_corr_plotter(self,sub3,F1cor_norm,F2cor_norm,labels[i]) #plot corrected spectra variance:excitation dependent
            unmix_corr_plotter(self,sub4,D1norm,D2norm,A1cor_norm,A2cor_norm,labels[i])#unmixed components of the corrected spectra
        
    
    def boxplot(self,param='Eff',ex=2,observation='region',plot_by='group',
                lim_y=[0,1],specify=[],colors=[],rotate_labels=90):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)

        fig, axes = plt.subplots(nrows=1, ncols=1)
        bp=axes.boxplot(data,showfliers=True,widths=.5,notch=False,vert=True,
                        patch_artist=True);

        p=0
        for patch, in zip(bp['boxes']):
            patch.set_facecolor(colors[p])
            patch.set_alpha(.25)
            p=p+1

        axes.set_ylim(lim_y[0],lim_y[1])
        axes.set_ylabel(self.alias[param]+add_label)
        xticks(range(1,len(labels)+1),labels,rotation=rotate_labels);
        
    def significance(self,param='Eff',ex=2,observation='region',plot_by='group',
                threshold=.05,test="T-Test",print_pval=True,specify=[],colors=[]):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)
        
        stat_options=[f_oneway,mannwhitneyu,ttest_ind,kruskal]#,self.exact_mc_perm_test]
        stat_label=["One Way ANOVA","Mann-Whitney U Test","T-Test","Kruskal-Wallis Test"]#,"Exact_Permutation_Test"]
        
        if test=='All':        
            stat_type=stat_options        
        else:
            stat_type=[stat_options[stat_label.index(test)]]
        
        pvals=np.zeros((len(stat_type),len(data),len(data)))
        
        for t in range(len(stat_type)):
            for i in range(len(data)):
                for j in range(len(data)):
                    pvals[t,i,j]=stat_type[t](data[i],data[j])[1]
                    
        if print_pval==True:
            print(pvals)
            
        for i in range(len(pvals)):
            t_image=(pvals[i]<threshold)
            matshow(t_image,interpolation="none",cmap="Greys",vmax=1,vmin=0)
            xticks(range(len(labels)),labels,rotation=90);
            yticks(range(len(labels)),labels);
            cbar=colorbar(label=stat_label[stat_options.index(stat_type[i])])
            cbar.set_ticks([0,1])
            cbar.set_ticklabels(["p-value>"+str(threshold),"p-value<"+str(threshold)])
            figure()

        
    def export_xlsx(self,observation=['group','image','region','pix']):

        writer = pd.ExcelWriter('Data_Archive/'+self.ExpName+'.xlsx')
        self.info.T.to_excel(writer,'Info')
        for o in range(len(observation)):
            DF=getattr(self,(observation[o]+'_DF')).copy()
            DF.to_excel(writer,observation[o])

        writer.save()
        
    def import_xlsx(self,FN,observation=['group','image','region','pix']):
        excel=pd.read_excel(FN,sheetname=None)
        self.info=excel['Info']
        self.calibFromInfo()
        if 'group' in observation:
            self.group_DF=excel['group']
        if 'image' in observation:
            self.image_DF=excel['image']
        if 'region' in observation:
            self.region_DF=excel['region']
        if 'pix' in observation:
            self.pix_DF=excel['pix']
        

class FRET_Calib:

    def __init__(self,D_name='Cerulean',A_name='Venus',FS_name="C5V",Qd=.62,Qa=.57,
                 ex1=405,ex2=458,
                 em_start=416,em_stop=718,em_channels=32):
        """Initializes calibration
        """        
        now=datetime.datetime.now()
        self.date=str(now.month)+'-'+str(now.day)+'-'+str(now.year)    
        self.fluor_names=[D_name,A_name]        
        self.ex_names=[str(ex1),str(ex2)]
        self.FS_name=FS_name
        self.Qd=Qd
        self.Qa=Qa
        self.ex1=ex1
        self.ex2=ex2
        self.wl_em=np.linspace(em_start,em_stop,em_channels)
        self.channels=em_channels
        self.CalibName=('Calib_'+D_name+'_'+A_name+'_'+
                        str(ex1)+'-'+str(ex2)+'_'+self.date)

    
        #dictionary relating short form attribute names to their long form
        self.alias={'I':'Intensity','Mx':'Spectra Maximum',
                    'D':self.fluor_names[0]+' Magnitude',
                    'A':self.fluor_names[1]+' Magnitude',
                    'AF':'Autofluorescence Magnitude',
                    'R':'Normalized Residual',
                    'Alpha':'Alpha: '+self.fluor_names[0]+'('+self.ex_names[1]+' / '+self.ex_names[0]+')',
                    'Beta':'Beta: '+self.fluor_names[1]+'('+self.ex_names[1]+' - Alpha * '+self.ex_names[0]+')',
                    'A_dir':self.fluor_names[1]+' Direct Excitation',
                    'Eff':'FRET Efficiency',
                    'D_ratio':'Donor Ratio: ('+self.ex_names[0]+'/'+self.ex_names[1]+')',
                    'A_ratio':'Acceptor Ratio: ('+self.ex_names[0]+'/'+self.ex_names[1]+')',
                    'eD_eA':self.fluor_names[0]+' to '+self.fluor_names[1]+' excitation ratio',
                    'Eff_S':'Stoicheometry Corrected FRET Efficiency',
                    'S':'Stoicheometry (['+self.fluor_names[0]+']/['+self.fluor_names[1]+'])',
                    'Ind':'FRET Index',
                    'group':'Group Number','image':'Image Number','region':'Region Number',
                    'x_loc':'X Coordinate','y_loc':'Y Coordinate','area':'Observation Area (Pixels)'}
        
        #dictionary relating long form attribute names to their short form
        #(reverses mapping on self.alias)
        self.alias_inv={v: k for k, v in self.alias.iteritems()}
        
        
        #list of parameters which are excitation independent
        self.ex_independent=['area','group','image','region','x_loc','y_loc',
                             'D_ratio','A_ratio','Alpha','Beta','Eff','Eff_S',
                             'S','group','image','region']
                     
        #colorlist for plotting               
        self.clist=['blue','green','red','purple','orange','magenta','teal',
                    'cyan','olive','darkred','lime','goldenrod','black',
                    'coral','saddlebrown','lightskyblue','steelblue',
                    'grey','peru','palevioletred','tan']



#---------------------------------------------
#
#FUNCTIONS FOR READING AND PRE-PROCESSING IMAGES
#
#---------------------------------------------



    def get_nested_subdirectories(self,print_names=False):
        """Recursively reads directories in 'Calibration Inputs' folder 
           to get all the image filenames for importing
            
            Directories must be structured as such:
            
            Calibration Inputs
                Acceptor
                    Frequency 1
                        F1 tifs (multi channel spectral images)
                    Frequency 2
                        F2 tifs (multi channel spectral images)
                    Masks
                        Mask tifs (single channel binary images)
                AutoF...
                
            then generates the variables self.group_names and self.file_names
            (format: self.group_names[group],dtype=string)
            (format: self.file_names[group][F1,F2,Mask][image],dtype=string)
        """
        
        #get group directory list
        root_dir=os.getcwd()+'/CalibrationFiles/Calibration_Inputs/'        
        g_dir=[]
        for (dirname,dirs,files) in os.walk(root_dir):
            g_dir.append(dirs)
        
        #get file names
        FN=[]
        FN2=[]
        for i in range(len(g_dir[0])):
            t=[]
            t2=[]
            for (dirname,dirs,files) in os.walk(os.path.join(root_dir,g_dir[0][i])):
                flist=[]
                flist2=[]
                for filename in files:
                    if filename.endswith(".tif"):
                        flist.append(os.path.join(dirname,filename))
                        if print_names==True:
                            print(filename)
                        #extracts the image name from filepath
                        flist2.append(filename.split("/")[-1].split(".")[0])
                        
                #removes empty list creation that results from os.walk function
                if len(flist)>0:
                    t.append(flist)
                    t2.append(flist2)
            FN.append(t)
            FN2.append(t2)
            
        self.group_names=g_dir[0]
        self.file_names=FN
        #Keeps track of the index of each group for calculations below
        n=np.array(self.group_names)
        ind_array=arange(len(n))
        
        self.group_index_donor=ind_array[n=="Donor"][0]
        self.group_index_acceptor=ind_array[n=="Acceptor"][0]
        
        if "AutoF" in self.group_names:        
            self.group_index_AutoF=ind_array[n=="AutoF"][0]
            self.no_AutoF=False
        else:
            self.no_AutoF=True
        
        if "FRET_Standard" in self.group_names:
            self.group_index_FRET_Standard=ind_array[n=="FRET_Standard"][0]
            self.no_FRETStandard=False
        else:
            self.no_FRETStandard=True
            
        self.image_names=FN2
        
        
    def get_tif_container(self,interpolation=False):
        """Imports Tiff data into numpy arrays based on the filenames determined 
            from 'self.get_nested_subdirectories()' and generates the 
            self.tif_array variable
             (format: self.tif_array[group][F1,F2,Mask][image,em,x,y]or[image,x,y](for mask),dtype=float)
            
        """
        
        FN=self.file_names
        #get size of the first image for array containers
        channels_i,x,y=tf.TiffFile(FN[0][0][0]).asarray().shape
        
        if interpolation!=False:
            x=int(interpolation[0])
            y=int(interpolation[1])
    
        if self.channels==[]:        
            self.channels=channels_i        
        if channels_i!=self.channels:
            print('Number of Channels in Image Do Not Match Specified Emission Channels. Could be due to extra transmission channel or incorrect specification of the number of emission channels. This will default to the specified number of emission channel')
        self.img_num=[]        
        self.X=x
        self.Y=y
        tif_container=[]
        group_num=[]        
        dim3=[]        
        for group in tnrange(len(FN)):
            #determines number of images in subdir group
            im_count=len(FN[group][0])
            self.img_num.append(im_count)
            #creates ndarray to store all images
            F1=np.zeros((im_count,self.channels,x,y))
            F2=np.zeros((im_count,self.channels,x,y))
            Mlabel=np.ones((im_count,x,y))
            image_num=[]            
            dim2=[]            
            for image in range(im_count):#loops through subdir list and groups all images into array container
                F1t=tf.TiffFile(FN[group][0][image]).asarray()[:self.channels,:,:]
                F2t=tf.TiffFile(FN[group][1][image]).asarray()[:self.channels,:,:]
                xi,yi=F1t.shape[1:]                
                
                for channel in range(self.channels):
                    if interpolation!=False:
                        F1[image,channel,:,:]=imresize(F1t[channel,:,:],size=(x,y),interp='bilinear',mode='F')
                        F2[image,channel,:,:]=imresize(F2t[channel,:,:],size=(x,y),interp='bilinear',mode='F')
                    else:
                        F1[image,channel,:,:]=F1t[channel,:,:]
                        F2[image,channel,:,:]=F2t[channel,:,:]
                
                M1t=np.ones((xi,yi))
                M2t=np.ones((xi,yi))
                
                if len(FN[group])>2:
                    M1t=tf.TiffFile(FN[group][2][image]).asarray()
                    M1t=(M1t/np.max(M1t))
                if len(FN[group])>3:
                    M2t=tf.TiffFile(FN[group][3][image]).asarray()
                    M1t=(M1t/np.max(M1t))

                Mt=M1t*M2t
                
                if interpolation!=False:
                    Mt=imresize(Mt,size=(x,y),interp='bilinear',mode='F')
                    Mt=(Mt/np.max(Mt))                    
                    Mt=np.rint(Mt)
                #imshow(Mt),colorbar()
                #figure()
                #print("true")
                Mlabel[image,:,:]=label(Mt)
                
    
                #loops through labeled masks and sums up the number of unique 
                #regions in the mask, these get appended to a list since they 
                #are variable in size. 
                region_num=[]
                for r in range(0,int(np.max(Mlabel[image,:,:]))):
                    p=np.sum(Mlabel[image]==(r+1))
                    region_num.append(p)
                image_num.append(region_num)
            group_num.append(image_num)
            #adds all ndimage arrays into a list
            tif_container.append([F1,F2,Mlabel])
    
        #tif_container[exp groups][F1,F2,Mask][img,ch,x,y] or [img,x,y](for Mask)    
        self.shape=group_num
        self.tif_array=tif_container
        
    
    def register_images(self,to_F1=False):
        """Registers the images from the two excitation frequencies and updates
            self.tif_array variable with the shifted images
            (format: self.tif_array[group][F1,F2,Mask][image,em,x,y]or[image,x,y](for mask),dtype=float)
 
             Keyword Arguments:
             
           'to_F1=bool' option registers frequency 2 image to the frequency 1 image
            as opposed to registering frequency 1 to frequency 2 (default)
        """
        #tif_container[exp groups][F1,F2,Mask][img,ch,x,y] or [img,x,y](for Mask)    
        #registers images to account for image drift (F1 is translated by default)
        tif_container=self.tif_array
        
        m_ind=0 #mobile index
        s_ind=1 #stationary index
        if to_F1==True:
            m_ind=1
            s_ind=0
        
        for g in tnrange (len(tif_container)):
            for i in range (tif_container[g][0].shape[0]):
                m_im=np.sum(tif_container[g][m_ind][i,:,:,:],axis=0)
                s_im=np.sum(tif_container[g][s_ind][i,:,:,:],axis=0)
                yshift,xshift=register_translation(s_im,m_im)[0]
                print('Y: '+str(format(yshift,'1.3f')),'X: '+str(format(xshift,'1.3f')))
                lg_shift=(abs(yshift)>(0.05*self.Y))or(abs(xshift)>(0.05*self.X))                 
                
                if lg_shift:
                    check=eval(raw_input('Register using this large registration shift? (True or False)'))
                    if check:
                        for em in range(tif_container[g][0][0,:,:,:].shape[0]):
                            tif_container[g][m_ind][i,em,:,:]=transform_img(tif_container[g][m_ind][i,em,:,:],
                                                                            scale=1.0,angle=0.0,
                                                                            tvec=(yshift,xshift))
                else:
                    for em in range(tif_container[g][0][0,:,:,:].shape[0]):
                        tif_container[g][m_ind][i,em,:,:]=transform_img(tif_container[g][m_ind][i,em,:,:],
                                                                        scale=1.0,angle=0.0,
                                                                        tvec=(yshift,xshift))
            print('_____________')
        self.tif_array=tif_container


#---------------------------------------------
#
#FUNCTIONS FOR ORGANIZING AND ANALYZING DATA
#
#---------------------------------------------

        
    
    def sort_tif_container(self,central_tendency='median'): 
        """Sorts and averages pixels into DataFrames according to observation type 
        (pix_DF,region_DF,image_DF and groupDF)
        
        central_tendency parameter chooses either the mean or the median 
        to find the average spectra. This setting also sets the default variance 
        measurement, standard deviation (SD) for the mean and median average 
        deviation (MAD) for the median.
            
        """
        
        pixels=0
        regions=0
        images=0
        groups=len(self.shape)
        
        #self.shape is a series of nested lists with variable dimensions. The code below parses the total counts of each dimension
        
        # overall container shape, corresponds to number of directories in ~/Calibration_Inputs/
        for g in range (0,len(self.shape)):
            #Number of images in Sub-directory within ~/Calibration_Inputs/sub_dir/, where sub_dir could be [Donor, Acceptor, FRET_Standard,AutoF]
            images=images+len(self.shape[g])
            for i in range (0,len(self.shape[g])):
                #Number of unique regions within a mask generated by the label() function from skimage.measure 
                regions=regions+len(self.shape[g][i])                
                for r in range (0,len(self.shape[g][i])):
                    pixels=pixels+self.shape[g][i][r]

        print('Total Pixels: '+str(pixels))
        print('Total Regions: '+str(regions))
        print('Total Images: '+str(images))
        print('Total Groups: '+str(groups))
        self.counts=[groups,images,regions,pixels]
        
        #sets the central tendency and error functions to be used
        if central_tendency=='median':
            self.ct_name='median'        
            self.ct_function=np.nanmedian
            self.ct_err_name='MAD'        
            self.ct_err_function=robust.mad
        if central_tendency=='mean':
            self.ct_name='mean'        
            self.ct_function=np.nanmean
            self.ct_err_name='SD'        
            self.ct_err_function=np.nanstd
        
        [groups,images,regions,pixels]=self.counts
        #Groups refers to the number of experimental group sub-directories in 'Grouped_Tiffs' folder
        group_container=np.full([groups,3,self.channels],np.nan)
        
        #Images refer to the number of linked images that make up an image observation: Image1 refers to [Image1_Freq1,Image1_Freq2,Image1_Mask]:
        image_container=np.full([images,3,self.channels],np.nan)
        
        #regions refer to the number of linked regions that make up a region observation: Region1 refers to [Region1_Freq1,Region1_Freq2,Region1_Mask]:
        region_container=np.full([regions,3,self.channels],np.nan)
        
        #pixel refers to number of linked pixels that make up a pixel observation: Pixel1 refers to [Pixel1_Freq1, Pixel1_Freq2,Pixel_Mask]
        pix_container=np.full([pixels,3,self.channels],np.nan)
        pix=0
        reg=0
        img=0
        grp=0
        
        pix_labels=[]        
        reg_labels=[]
        image_labels=[]
        group_labels=[]
        
        for g in tnrange (0,len(self.shape)):
            g_start=pix
            g_name=self.group_names[g]
            group_labels.append(g_name)
            g_img_count=0
            g_reg_count=0
            for i in range (0,len(self.shape[g])):
                i_start=pix
                i_name=[g_name,g_name+': '+self.image_names[g][0][i].replace(self.ex_names[0],'')]
                image_labels.append(i_name)
                i_reg_count=0
                for r in range (1,len(self.shape[g][i])+1):
                    r_start=pix
                    r_name=[g_name,i_name[1],i_name[1]+': R-'+str(r)]
                    reg_labels.append(r_name)
                
                    mask_coor = np.nonzero(self.tif_array[g][2][i,:,:]==r)
                    for (x,y) in zip(*mask_coor):
                        p_name=[g_name,i_name[1],r_name[2],r_name[2]+': ('+str(x)+','+str(y)+')']
                        pix_labels.append(p_name)
                        pix_container[pix,0,:self.channels]=self.tif_array[g][0][i,:,x,y]
                        pix_container[pix,1,:self.channels]=self.tif_array[g][1][i,:,x,y]
                        pix_container[pix,2,0:6]=g,i,r,x,y,1
                        pix=pix+1
                       
                    region_container[reg,:,:self.channels]=self.ct_function(pix_container[r_start:pix,:,:self.channels],axis=0)
                    region_container[reg,:,:self.channels]=self.ct_function(pix_container[r_start:pix,:,:self.channels],axis=0)
                    region_container[reg,2,5]=(pix-r_start)
                    reg=reg+1
                    g_reg_count=g_reg_count+1
                    i_reg_count=i_reg_count+1
                image_container[img,:,:self.channels]=self.ct_function(pix_container[i_start:pix,:,:self.channels],axis=0)
                image_container[img,:,:self.channels]=self.ct_function(pix_container[i_start:pix,:,:self.channels],axis=0)
                image_container[img,2,5]=(pix-i_start)
                image_container[img,2,2]=i_reg_count               
                img=img+1
                g_img_count=g_img_count+1
            group_container[grp,:,:self.channels]=self.ct_function(pix_container[g_start:pix,:,:self.channels],axis=0)
            group_container[grp,:,:self.channels]=self.ct_function(pix_container[g_start:pix,:,:self.channels],axis=0)
            group_container[grp,2,5]=(pix-g_start)
            group_container[grp,2,2]=g_reg_count
            group_container[grp,2,1]=g_img_count
            grp=grp+1
        #subroutine to convert array to a labeled Dataframe
        def container_to_DF(container,labels):
            
            dat={'spectra1':container[:,0,:].tolist(),
                 'spectra2':container[:,1,:].tolist(),
                 'group':container[:,2,0],
                 'image':container[:,2,1],
                 'region':container[:,2,2],
                 'y_loc':container[:,2,3],
                 'x_loc':container[:,2,4],
                 'area':container[:,2,5]}
            DF=pd.DataFrame(data=dat)                        
            DF.insert(0,'label',labels)
            return(DF)
            
        self.pix_DF=container_to_DF(pix_container,pix_labels)
        self.region_DF=container_to_DF(region_container,reg_labels)
        self.image_DF=container_to_DF(image_container,image_labels)
        self.group_DF=container_to_DF(group_container,group_labels)
        
        
    def create_components(self):
        """ Creates the normalized spectra used for the spectral unmixing
        steps
        """
        
        #Average spectra are defined as the overall mean spectra of group such that unfitting components are derived from
        #Maximum number of pixels
        D1ave=self.group_DF.loc[self.group_index_donor].spectra1
        D2ave=self.group_DF.loc[self.group_index_donor].spectra2
        A1ave=self.group_DF.loc[self.group_index_acceptor].spectra1
        A2ave=self.group_DF.loc[self.group_index_acceptor].spectra2
        
        if self.no_AutoF==True:
            AF1ave=np.zeros(np.shape(D1ave))
            AF2ave=np.zeros(np.shape(D1ave))
        else:
            AF1ave=self.group_DF.loc[self.group_index_AutoF].spectra1
            AF2ave=self.group_DF.loc[self.group_index_AutoF].spectra2
        
        D1norm=D1ave/np.trapz(D1ave)
        D2norm=D2ave/np.trapz(D2ave)
        A1norm=A1ave/np.trapz(A1ave)
        A2norm=A2ave/np.trapz(A2ave)
        
        if np.trapz(AF1ave)!=0:        
            AF1norm=AF1ave/np.trapz(AF1ave)
            AF2norm=AF2ave/np.trapz(AF2ave)
        else:
            AF1norm=AF1ave
            AF2norm=AF2ave
        
        figure()
        plot(self.wl_em,D1norm,label=self.fluor_names[0]+' '+self.ex_names[0]+'nm',color='blue',ls='--')
        plot(self.wl_em,D2norm,label=self.fluor_names[0]+' '+self.ex_names[1]+'nm',color='blue',ls='-')
        plot(self.wl_em,A1norm,label=self.fluor_names[1]+' '+self.ex_names[0]+'nm',color='orange',ls='--')
        plot(self.wl_em,A2norm,label=self.fluor_names[1]+' '+self.ex_names[1]+'nm',color='orange',ls='-')
        plot(self.wl_em,AF1norm,label='AutoF '+self.ex_names[0]+'nm',color='purple',ls='--')
        plot(self.wl_em,AF2norm,label='AutoF '+self.ex_names[1]+'nm',color='purple',ls='-')
        legend()
        xlabel('Wavelength (nm)')
        ylabel('Normalized Intensity')
        show(block=False)
        
        self.multicomp=eval(raw_input('Fit using excitation dependent components? (True or False) '))
        
        if self.multicomp==False:
            Donor_freq=int(raw_input('Which excitation should determine donor shape? (1='+self.ex_names[0]+'nm,2='+self.ex_names[1]+'nm) '))
            Acceptor_freq=int(raw_input('Which excitation should determine acceptor shape? (1='+self.ex_names[0]+'nm,2='+self.ex_names[1]+'nm) '))
         
            if Donor_freq==1:
                D2norm=D1norm
            if Donor_freq==2:
                D1norm=D2norm

            if Acceptor_freq==1:
                A2norm=A1norm
            if Acceptor_freq==2:
                A1norm=A2norm
        
        
        self.comps1=np.array([D1norm,A1norm,AF1norm])
        self.comps2=np.array([D2norm,A2norm,AF2norm])
        
        


    def unmix_DF(self,observation='all'):
        """Cycles through a DataFrame and unmixes the spectra into different 
        fitting components that are then saved into the DataFrame
        
        observation='all' option unmixes all dataframes or a list of observations
        types can be specified, for example: observation=["group","image",region]        
        """
        if observation=='all':
            obs=['group','image','region','pix']
        else:
            obs=observation
        
        self.DF_names=[]
        
        for o in obs:
            self.DF_names.append(o+"_DF")
            oDF=o+'_DF'
            DF=getattr(self,oDF)
            spectra1=DF.spectra1.values
            spectra2=DF.spectra2.values
            params=[]      
            for i in tnrange(len(DF.index),desc=o+" Loop"):
                #print(spectra1[i])
                #print(spectra2[i])
                params.append(self.unmix_magnitudes(spectra1[i],spectra2[i],self.comps1,self.comps2))
            params=np.array(params).T
            #params_list must coorespond to the order of returned parameters from unmix_magnitudes
            param_list=['D_D1','D_AF1','D_R1','D_D2','D_AF2','D_R2',
                         'A_A1','A_AF1','A_R1','A_A2','A_AF2','A_R2',
                         'F_D1','F_A1','F_AF1','F_R1',
                         'F_D2','F_A2','F_AF2','F_R2',
                         'I1','I2']
            
            for p in range(len(param_list)):        
                DF[param_list[p]]=params[p]
                
                
    def unmix_magnitudes(self,spectra1,spectra2,comps1,comps2):
        """Takes paired spectra and unmixing matrices from excitations 1 and 2 
        as input. Returns an object containing all magnitudes needed for any 
        processing"""
        
        #unpack normalized components for fitting
        dcomp1,acomp1,afcomp1=comps1[0],comps1[1],comps1[2]
        dcomp2,acomp2,afcomp2=comps2[0],comps2[1],comps2[2]
        
        #remove filled nan values at ends of array
        spectra1=np.nan_to_num(spectra1[:self.channels])
        spectra2=np.nan_to_num(spectra2[:self.channels])
        
        #Compute Area under spectra, integrated intensity
        I1=np.trapz(spectra1)
        I2=np.trapz(spectra2)
        
        #Unmix Donor Magnitudes or Acceptor Magnitudes for use in gamma from single labeled samples
        [DD1,DAF1],DR1=unmix(np.array([dcomp1,afcomp1]).T,spectra1)
        [DD2,DAF2],DR2=unmix(np.array([dcomp2,afcomp2]).T,spectra2)
        [AA1,AAF1],AR1=unmix(np.array([acomp1,afcomp1]).T,spectra1)
        [AA2,AAF2],AR2=unmix(np.array([acomp2,afcomp2]).T,spectra2)
        
        #Unmix Donor,Acceptor,AF signal Magnitudes for FRET spectra, denoted with F_
        [FD1,FA1,FAF1],FR1=unmix(comps1.T,spectra1)
        [FD2,FA2,FAF2],FR2=unmix(comps2.T,spectra2)
        
        return([DD1,DAF1,DR1/I1,
                DD2,DAF2,DR2/I2,
                AA1,AAF1,AR1/I1,
                AA2,AAF2,AR2/I2,
                FD1,FA1,FAF1,FR1/I1,
                FD2,FA2,FAF2,FR2/I2,
                I1,I2])

    def new_calculation(self,calc,name=False):
        """Allows new parameters to be created based off of simple operations,
        for example the normalized donor magnitude can be generated by:
        
        self.new_calculation(['D1','/','I1'],name='Dnorm_1')
        
        
        """
        operators= {'+': (lambda x,y: x+y),
                    '-': (lambda x,y: x-y),
                    '*': (lambda x,y: x*y),
                    '/': (lambda x,y: x/y),
                    '^': (lambda x,y: x**y)}
        if name==False:
            name=calc[0]+calc[1]+calc[2]
        
        try:
            self.pix_DF[name]=operators[calc[1]](
                             getattr(self.pix_DF,calc[0]),
                             getattr(self.pix_DF,calc[2]))
        except:
            print('Pix Calculation Failed')
        try:
            self.region_DF[name]=operators[calc[1]](
                             getattr(self.region_DF,calc[0]),
                             getattr(self.region_DF,calc[2]))
        except:
            print('Region Calculation Failed') 
        try:
            self.image_DF[name]=operators[calc[1]](
                             getattr(self.image_DF,calc[0]),
                             getattr(self.image_DF,calc[2]))
        except:
            print('Image Calculation Failed')
        try:
            self.group_DF[name]=operators[calc[1]](
                             getattr(self.group_DF,calc[0]),
                             getattr(self.group_DF,calc[2]))
        except:
            print('Group Calculation Failed') 
            
    def default_calculations(self):
        """Generates the standard parameters needed for the FRET analysis
            ie. alpha,beta,direct excitation, and efficiency
        """
        #Indicies needed for calculating the calibration constants
        Di=self.group_index_donor
        Ai=self.group_index_acceptor
        
        #Donor and Acceptor Magnitudes from Single Labeled Samples unmixed from autofluorescence
        D1=self.group_DF.D_D1[Di]
        D2=self.group_DF.D_D2[Di]
        A1=self.group_DF.A_A1[Ai]
        A2=self.group_DF.A_A2[Ai]
        
        self.gamma=(D2/D1)*(A1/A2)
        print('Gamma: '+str(self.gamma))
        #Cycles through each observation dataframe
        #DF_names=['pix_DF','region_DF','image_DF','group_DF']
        for df in self.DF_names:
            DF=getattr(self,df)
            
            DF['D_ratio']=DF.D_D1/DF.D_D2
            DF['A_ratio']=DF.A_A1/DF.A_A2
            
            DF['Ind1']=DF.F_A1/DF.F_D1
            DF['Ind2']=DF.F_A2/DF.F_D2
            
            DF['Alpha']=DF.F_D2/DF.F_D1
            DF['Beta']=DF.F_A2-DF.Alpha*DF.F_A1
            
            DF['A_dir1']=DF.Beta/(DF.Alpha*(self.gamma**-1-1))
            DF['A_dir2']=DF.Beta/(1-self.gamma)
            
            DF['Eff']=(((DF.F_D2*self.Qa)/((DF.F_A2-DF.A_dir2)*self.Qd))+1)**(-1)
             
            DF['eD_eA1']=(DF.F_D1/self.Qd+(DF.F_A1-DF.A_dir1)/self.Qa)/(DF.A_dir1/self.Qa)
            DF['eD_eA2']=(DF.F_D2/self.Qd+(DF.F_A2-DF.A_dir2)/self.Qa)/(DF.A_dir2/self.Qa)

        #The Donor to Acceptor extinction coefficient ratios at each frequency 
        #assuming the stoicheometry of the FRET standard is 1:1
        if self.no_FRETStandard==True:
            self.exRatio1=np.nan
            self.exRatio2=np.nan
        else:        
            Fi=self.group_index_FRET_Standard            
            self.exRatio1=self.group_DF.eD_eA1[Fi]
            self.exRatio2=self.group_DF.eD_eA2[Fi]
            
        print('eD/eA (freq1): '+str(self.exRatio1))
        print('eD/eA (freq2): '+str(self.exRatio2))
        
        for df in self.DF_names:
            DF=getattr(self,df)
            
            DF['S']=DF.eD_eA2*self.exRatio2**-1
        
            DF['Eff_S']=(DF.F_A2-DF.A_dir2)/DF.A_dir2*self.exRatio2**-1
        



#---------------------------------------------
#
#SUBROUTINES FOR PLOTTING FUNCTIONS
#
#---------------------------------------------



    def specify_mask(self,DF,s):
        """Excludes data from DataFrame unless it meet the conditions fed in 
        from s.
        
        s format: 
        ['parameter','logic operator', [list of values(can be length 1)]]
        """
        
        logic={"<": (lambda x,y: x<y[0]), 
               ">": (lambda x,y: x>y[0]),
               "><": (lambda x,y: (x>y[0])&(x<y[1])),
               "=": (lambda x,y: x.isin(y)),
               "!=": (lambda x,y: -x.isin(y))}
        #    chosen function (DF.parameter,value list)
        mask=logic[s[1]] (getattr(DF,s[0]),s[2])
            
        return(DF[mask])
    
    def get_figure_sets(self, DF, param, specify, plot_by='group'):
        """ Gets the data and labels for a plotting function from a observation 
        dataframe based on the parameter requested, the type of plot, and any 
        conditional statements in specify
        """
       
       
        if plot_by=='group':
            cutoff=0
        if plot_by=='image':
            cutoff=1
        if plot_by=='region':
            cutoff=2
            
        
        DF_masked=DF.copy()
        for s in specify:
            DF_masked=self.specify_mask(DF_masked,s)
        
        dat=np.array(getattr(DF_masked,param).values)
        labels=np.vstack(DF_masked.label.values)
        temp_labels=[]
        for obs in range(len(labels)):
            #appends location id up to the cutoff determined by plot_by
            temp_labels.append(str(labels[obs][cutoff]))
        temp_labels=np.array(temp_labels)
        fig_labels=np.unique(temp_labels)
        
        Data=[]
        for n in fig_labels:
            Data.append(dat[temp_labels==n])
        return(Data,fig_labels)    


#---------------------------------------------
#
#PLOTTING FUNCTIONS
#
#---------------------------------------------
        
    def spectraVariance(self,ex=1,observation='region',plot_by='group',
                        specify=[],colors=[]):
        """Plots the noise characteristics of the spectra. First panel shows the
        average spectra with a filled band 
        """
        def genSpectraplot(self,Spectra_array,sub1,sub2,sub3,sub4,l,c):
            Spectra=Spectra_array.T/np.trapz(Spectra_array,axis=1)
            Spectra_ave=np.nanmean(Spectra,axis=1)
            Spectra_stdev=np.nanstd(Spectra,axis=1)
            N=Spectra.shape[1]
            sub1.plot(self.wl_em,Spectra_ave,color=c,ls='--',label=l+' (n='+str(N)+')')
            sub1.fill_between(self.wl_em,(Spectra_ave+Spectra_stdev),(Spectra_ave-Spectra_stdev),color=c,alpha=.25)
            sub1.set_xlabel('Wavelength')
            sub1.set_ylabel('Normalized Intensity')
            
            sub3.plot(self.wl_em,Spectra_stdev/(Spectra_ave**.5),ls='',marker='o',color=c)
            sub3.set_xlabel('Wavelength')
            sub3.set_ylabel('Slope')
            
            sub2.plot((np.mean(Spectra_array,axis=0)**.5),Spectra_stdev,ls='',marker='o',color=c)
            sub2.set_ylabel('Standard Deviation')
            sub2.set_xlabel('Intensity^.5')
            
            sub4.axis('off')            
            
            sub1.legend(loc='upper left', bbox_to_anchor=(1.18,-.18))        
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        param_ex='spectra'+str(ex)
        add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)
              
        Varfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=[15,16])
        
        for i in range(len(data)):
            genSpectraplot(self,np.vstack(data[i]),sub1,sub2,sub3,sub4,
                          labels[i]+add_label,colors[i%len(colors)]) 
        
    
    def showImage(self,param='Eff',ex=2,mn=0,mx=1,specify=[]):
        
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=self.pix_DF.copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by='image')
        xloc,labels=self.get_figure_sets(DF,'x_loc',specify,plot_by='image')
        yloc,labels=self.get_figure_sets(DF,'y_loc',specify,plot_by='image')
        
        for i in range(len(data)):
            figure()
            image=np.full([self.X,self.Y],np.nan)
            imshow(np.ones([self.X,self.Y]),cmap='Greys_r')
            for p in range(len(data[i])):
                image[int(yloc[i][p]),int(xloc[i][p])]=data[i][p]
            imshow(image,vmin=mn,vmax=mx),colorbar(label=self.alias[param]+add_label)
            title(labels[i]+add_label)
            axis('off')
            
            
    def plotHistogram(self,param='D',ex=2,lim_x=[0,1],bins=100,norm=False,
                      observation='region',plot_by='group',specify=[],colors=[]):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)

        for i in range(len(data)):
            axvline(self.ct_function(data[i]), color=colors[i%len(colors)],
                    ls='--',label=labels[i]+' ('+self.ct_name+')')
            hist(data[i],histtype='stepfilled',alpha=.5,color=colors[i%len(colors)],
                 bins=bins,range=(lim_x[0],lim_x[1]),normed=norm) 
            
        legend()
        xlabel(self.alias[param]+add_label)
        ylabel('Frequency')
        
    def plotScatter(self,param_x='D',ex_x=2,param_y='D',ex_y=2,
                    lim_x=[0,1],lim_y=[0,1],observation='region',plot_by='group',
                    specify=[],colors=[],alpha=1,fit=False,print_fit=False,ls=''):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param_x in self.alias_inv:
            param_x=self.alias_inv[param_x]
        
        if param_y in self.alias_inv:
            param_y=self.alias_inv[param_y]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param_x in self.ex_independent:
            param_x_ex=param_x            
            add_label_x=''
        else:
            param_x_ex=param_x+str(ex_x)
            add_label_x=' ('+self.ex_names[ex_x-1]+')'
        
        if param_y in self.ex_independent:
            param_y_ex=param_y            
            add_label_y=''
        else:
            param_y_ex=param_y+str(ex_y)
            add_label_y=' ('+self.ex_names[ex_y-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data_x,labels=self.get_figure_sets(DF,param_x_ex,specify,plot_by=plot_by)
        data_y,labels=self.get_figure_sets(DF,param_y_ex,specify,plot_by=plot_by)
        
        for i in range(len(data_x)):
            plot(data_x[i],data_y[i],ls=ls,marker='o',alpha=alpha,
                 color=colors[i%len(colors)],label=labels[i]) 
            if fit==True:
                f=linregress(data_x[i],data_y[i])
                x_fit=np.linspace(np.nanmin(data_x[i]),np.nanmax(data_x[i]),100)
                y_fit=f[0]*(x_fit)+f[1]
                plot(x_fit,y_fit,color=colors[i%len(colors)],ls='--',label='Fit')
                if print_fit==True:
                    print(labels[i])
                    print('Fit Slope='+str(format(f[0],'1.3f')))
                    print('Fit Intercept='+str(format(f[1],'1.3f')))
                    print('Fit R^2='+str(format(f[2]**2,'1.3f')))
                    print('Fit P-Value='+str(format(f[3],'1.3f')))
        legend()
        xlabel(self.alias[param_x]+add_label_x)
        ylabel(self.alias[param_y]+add_label_y)
        xlim(lim_x[0],lim_x[1])
        ylim(lim_y[0],lim_y[1])
        
    def plotFRET(self,observation='region',plot_by='group', specify=[],print_params=False):
        
        def genFRETplot(self,F1,F2,name,print_params=False):
            """Generates the standard four figure plot that shows the calculation
            steps. Used within self.calcE_by(group/image/region)
            
            """    
            
            [D1,A1,AF1]=unmix(self.comps1.T,F1)[0]
            [D2,A2,AF2]=unmix(self.comps2.T,F2)[0]
    
            alpha=D2/D1
            beta=A2-alpha*A1
            A1_dir=beta/(alpha*(self.gamma**-1-1))
            A2_dir=beta/(1-self.gamma)
            Eff=(((D1*self.Qa)/((A1-A1_dir)*self.Qd))+1)**(-1)
            
            
            #Eff2=(((D2*self.Qa)/((A2-A2_dir)*self.Qd))+1)**(-1)
            
            FRETfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=(17,17))
            FRETfig.suptitle(name+' (E= '+'{0:.3f}'.format(Eff)+')',fontsize=30)
            
            sub1.set_title('Alpha Fit')
            sub1.plot(self.wl_em,F1,label='$F_{DA}^{1}$ Raw',ls='-',color='red')
            sub1.plot(self.wl_em,F2,label='$F_{DA}^{2}$ Raw',ls='--',color='red')
            sub1.plot(self.wl_em,D1*self.comps1[0,:],label='$F_{DA}^{1}$ Donor Fit',ls='-',color='blue',lw=3)
            sub1.plot(self.wl_em,D2*self.comps2[0,:],label='$F_{DA}^{2}$ Donor Fit',ls='--',color='blue',lw=3)
            sub1.plot(self.wl_em,AF1*self.comps1[2,:],label='$F_{DA}^{1}$ AF Fit',ls='-',color='green',lw=3)
            sub1.plot(self.wl_em,AF2*self.comps2[2,:],label='$F_{DA}^{2}$ AF Fit',ls='--',color='green',lw=3)
            sub1.set_xlabel('Wavelength')
            sub1.set_ylabel('Intensity')
            sub1.legend(loc='best')
            sub1.set_xlim([self.wl_em[0],self.wl_em[-1]])
            
            
            sub2.set_title('Beta Fit')
            sub2.plot(self.wl_em,((F2-AF2*self.comps2[2,:])-alpha*(F1-AF1*self.comps1[2,:])),c="blue",label='$F_{DA}^{2}- Alpha * F_{DA}^{1}$')
            sub2.plot(self.wl_em,self.comps2[1]*beta,c="red",ls='--',label='$\hat{e}_{A}$ * Beta',lw=4)
            sub2.set_xlabel('Emission Wavelength (nm)')
            sub2.set_ylabel('Intensity (au)')
            sub2.legend(loc='best')
            sub2.set_xlim([self.wl_em[0],self.wl_em[-1]])
            
            F1cor=F1-A1_dir*self.comps1[1]
            F2cor=F2-A2_dir*self.comps2[1]
        
            sub3.set_title('Subtraction of Direct Excitation')
            sub3.plot(self.wl_em,F1,ls='-',color='red',lw=2,label='$F_{DA}^{1}$ Raw')
            sub3.plot(self.wl_em,F2,ls='--',color='red',lw=2,label='$F_{DA}^{2}$ Raw')
            sub3.plot(self.wl_em,F1cor,ls='-',color='teal',lw=3,label='$F_{DA}^{1}$ Corrected')
            sub3.plot(self.wl_em,F2cor,ls='--',color='teal',lw=3,label='$F_{DA}^{2}$ Corrected')
            sub3.set_xlabel('Wavelength')
            sub3.set_ylabel('Intensity (au)')
            sub3.set_xlim([self.wl_em[0],self.wl_em[-1]])
            sub3.legend(loc='best',fontsize=12)
        
            sub4.set_title('Unmixed Components')
            sub4.plot(self.wl_em,F1cor-AF1*self.comps1[2,:],color='teal',ls='-',label='$F_{DA}^{1}$ Corrected',lw=2)
            sub4.plot(self.wl_em,self.comps1[0]*D1,color='blue',ls='-',label='$F_{DA}^{1}$ Donor Fit',lw=3)
            sub4.plot(self.wl_em,self.comps1[1]*(A1-A1_dir),color='orange',ls='-',label='$F_{DA}^{1}$ Acceptor Fit',lw=3)
            sub4.plot(self.wl_em,F2cor-AF1*self.comps2[2,:],color='teal',ls='--',label='$F_{DA}^{2}$ Corrected',lw=2)
            sub4.plot(self.wl_em,self.comps2[0]*D2,color='blue',ls='--',label='$F_{DA}^{2}$ Donor Fit',lw=3)
            sub4.plot(self.wl_em,self.comps2[1]*(A2-A2_dir),color='orange',ls='--',label='$F_{DA}^{2}$ Acceptor Fit',lw=3)
            sub4.set_xlabel('Wavelength')
            sub4.set_ylabel('Intensity (au)')
            sub4.set_xlim([self.wl_em[0],self.wl_em[-1]])
            sub4.legend(loc='best',fontsize=12)
            
        if observation=="pixel":
            observation="pix"
            
        DF=getattr(self,(observation+'_DF')).copy()
        
        spec1,labels=self.get_figure_sets(DF,'spectra1',specify,plot_by=plot_by)
        spec2,labels=self.get_figure_sets(DF,'spectra2',specify,plot_by=plot_by)
        
        
        for i in range(len(spec1)):
            ave_spec1=self.ct_function(np.vstack(spec1[i]),axis=0)
            ave_spec2=self.ct_function(np.vstack(spec2[i]),axis=0)
            genFRETplot(self,ave_spec1,ave_spec2,labels[i],print_params)
            
    def spectraFit(self,norm=True,observation='group',plot_by='group',specify=[],colors=[]):
         
        def plotter(self,F1,F2,D1,D2,A1,A2,AF1,AF2,l):
            #central tendency, could be mean or median
            F_ct1=self.ct_function(F1,axis=0)
            F_ct2=self.ct_function(F2,axis=0)
            F_var1=self.ct_err_function(F1,axis=0)
            F_var2=self.ct_err_function(F2,axis=0)
                   
            D1_ct=self.ct_function(D1,axis=0)
            D2_ct=self.ct_function(D2,axis=0)            
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            
            A1_ct=self.ct_function(A1,axis=0)
            A2_ct=self.ct_function(A2,axis=0)            
            A_var1=self.ct_err_function(A1,axis=0)
            A_var2=self.ct_err_function(A2,axis=0)
            
            AF1_ct=self.ct_function(AF1,axis=0)
            AF2_ct=self.ct_function(AF2,axis=0)            
            AF_var1=self.ct_err_function(AF1,axis=0)
            AF_var2=self.ct_err_function(AF2,axis=0)
            
            N=F1.shape[0]
            Varfig, (ax1,ax2) =subplots(1,2,figsize=[15,8])
            Varfig.suptitle(l +' (n='+str(N)+')',fontsize=25)
            
            ax1.plot(self.wl_em,F_ct1,color="red",ls='-',label= self.ct_name+" FRET "+self.ex_names[0])
            ax2.plot(self.wl_em,F_ct2,color="red",ls='-',label=self.ct_name+" FRET "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(F_ct1+F_var1),(F_ct1-F_var1),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(F_ct2+F_var2),(F_ct2-F_var2),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,D1_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,D2_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D1_ct+D_var1),(D1_ct-D_var1),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(D2_ct+D_var2),(D2_ct-D_var2),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,A1_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,A2_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(A1_ct+A_var1),(A1_ct-A_var1),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(A2_ct+A_var2),(A2_ct-A_var2),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,AF1_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[0])
            ax2.plot(self.wl_em,AF2_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(AF1_ct+AF_var1),(AF1_ct-AF_var1),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(AF2_ct+AF_var2),(AF2_ct-AF_var2),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            
            ax1.set_title(self.ex_names[0]+' nm')
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best',fontsize=12)
            
            ax2.set_title(self.ex_names[1]+' nm')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Normalized Intensity [Au]')
            ax2.legend(loc='best',fontsize=12)
        
        #get observation Dataframe
        if observation=="pixel":
            observation="pix"   
        DF=getattr(self,(observation+'_DF')).copy()
        
        #Generates a plot for each plot_by feature
        for i in tnrange(1):
            print("Getting Figure Sets")
            print("---------------------------------------------------------------------------------")
            #get filtered data (plot_by +specify filtering)
            spec1,labels=self.get_figure_sets(DF,"spectra1",specify,plot_by=plot_by)
            D1dat,labels=self.get_figure_sets(DF,"F_D1",specify,plot_by=plot_by)
            A1dat,labels=self.get_figure_sets(DF,"F_A1",specify,plot_by=plot_by)
            AF1dat,labels=self.get_figure_sets(DF,"F_AF1",specify,plot_by=plot_by)
            
            
            spec2,labels=self.get_figure_sets(DF,"spectra2",specify,plot_by=plot_by)
            D2dat,labels=self.get_figure_sets(DF,"F_D2",specify,plot_by=plot_by)
            A2dat,labels=self.get_figure_sets(DF,"F_A2",specify,plot_by=plot_by)
            AF2dat,labels=self.get_figure_sets(DF,"F_AF2",specify,plot_by=plot_by)
            
        print("Generating Plots")
        print("---------------------------------------------------------------------------------")
        for i in tnrange(len(spec1)):
            
            #unpack variables for computation
            D1,D2=D1dat[i],D2dat[i]
            A1,A2=A1dat[i],A2dat[i]
            AF1,AF2=AF1dat[i],AF2dat[i]
            F1=np.vstack(spec1[i])
            F2=np.vstack(spec2[i])
            
            comp_D1=np.tile(self.comps1[0],(len(D1),1))#converts comps array to shape of D1 for simple element wise matrix multiplication
            comp_D2=np.tile(self.comps2[0],(len(D2),1))#converts comps array to shape of D2 for simple element wise matrix multiplication
            comp_A1=np.tile(self.comps1[1],(len(A1),1))#converts comps array to shape of A1 for simple element wise matrix multiplication
            comp_A2=np.tile(self.comps1[1],(len(A2),1))#converts comps array to shape of A2 for simple element wise matrix multiplication
            comp_AF1=np.tile(self.comps1[2],(len(AF1),1))#converts comps array to shape of F1 for simple element wise matrix multiplication
            comp_AF2=np.tile(self.comps2[2],(len(AF2),1))#converts comps array to shape of F2 for simple element wise matrix multiplication
            
            #perform computations
            D1comp=(D1*comp_D1.T).T #Donor1 vectors
            D2comp=(D2*comp_D2.T).T #Donor2 vectors
            A1comp=(A1*comp_A1.T).T #Acceptor1 vectors
            A2comp=(A2*comp_A2.T).T #Acceptor2 vectors
            AF1comp=(AF1*comp_AF1.T).T #AutoF1 vectors
            AF2comp=(AF2*comp_AF2.T).T #AutoF2 vectors
            
            Mag1=np.trapz(F1,axis=1)
            Mag2=np.trapz(F2,axis=1)
            
            if norm==False:
                Mag1=np.ones(Mag1.shape)
                Mag2=np.ones(Mag2.shape)
            
            F1norm=(F1.T/Mag1).T
            F2norm=(F2.T/Mag2).T
            D1norm=(D1comp.T/Mag1).T
            D2norm=(D2comp.T/Mag2).T
            A1norm=(A1comp.T/Mag1).T
            A2norm=(A2comp.T/Mag2).T
            AF1norm=(AF1comp.T/Mag1).T
            AF2norm=(AF2comp.T/Mag2).T
            
            #call plotting functions 
            plotter(self,F1norm,F2norm,D1norm,D2norm,A1norm,A2norm,AF1norm,AF2norm,labels[i])
                
      
      
    def plotFRETvariance(self,observation='region',plot_by='group', specify=[],autoF=True):

        def alpha_plotter(self,ax1,F1,F2,D1,D2,label,color):
            #central tendency, could be mean or median
            F_ct1=self.ct_function(F1,axis=0)
            F_ct2=self.ct_function(F2,axis=0)
            F_var1=self.ct_err_function(F1,axis=0)
            F_var2=self.ct_err_function(F2,axis=0)
                   
            D1_ct=self.ct_function(D1,axis=0)
            D2_ct=self.ct_function(D2,axis=0)            
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            
            ax1.plot(self.wl_em,F_ct1,color="red",ls='-',label= self.ct_name+" FRET "+self.ex_names[0])
            ax1.plot(self.wl_em,F_ct2,color="red",ls='--',label=self.ct_name+" FRET "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(F_ct1+F_var1),(F_ct1-F_var1),color="red",alpha=.25,label="FRET $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(F_ct2+F_var2),(F_ct2-F_var2),color="red",alpha=.25)
            
            ax1.plot(self.wl_em,D1_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,D2_ct,color="blue",ls='--',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D1_ct+D_var1),(D1_ct-D_var1),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(D2_ct+D_var2),(D2_ct-D_var2),color="blue",alpha=.25)
            
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best',fontsize=12)
            ax1.set_title("Raw Spectra and Alpha")
            
            
        def Beta_plotter(self,ax1,beta_spectra,name):
            #normalize the spectra
            Spectra_ct=self.ct_function(np.nan_to_num(beta_spectra),axis=0)
            Spectra_var=self.ct_err_function(np.nan_to_num(beta_spectra),axis=0)
            
            #peak fitted acceptor component
            acceptor_ref=unmix(self.comps1.T,Spectra_ct)[0][1]*self.comps1[1]
            
            #acceptor_ref=np.trapz(Spectra_ct)*(self.comps1[1]/np.trapz(self.comps1[1]))
            ax1.plot(self.wl_em,Spectra_ct,color="green",ls='-',label=self.ct_name+' Beta')
            ax1.plot(self.wl_em,acceptor_ref,color="red",label=self.fluor_names[1]+" component",ls="--")
            ax1.fill_between(self.wl_em,(Spectra_ct+Spectra_var),(Spectra_ct-Spectra_var),color="green",alpha=.25,
                              label=self.ct_err_name+" $\pm$"+"1")
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.set_title("Beta")
            ax1.legend()
            
        def Spectra_corr_plotter(self,ax1,F1cor,F2cor,name):
            Spectra_ct1=self.ct_function(F1cor,axis=0)
            Spectra_ct2=self.ct_function(F2cor,axis=0)
    
            Spectra_var1=self.ct_err_function(F1cor,axis=0)
            Spectra_var2=self.ct_err_function(F2cor,axis=0)
    
            ax1.plot(self.wl_em,Spectra_ct1,color="purple",ls='-',label='Corrected FRET '+self.ex_names[0])
            ax1.plot(self.wl_em,Spectra_ct2,color="purple",ls='--',label='Corrected FRET '+self.ex_names[1])
            
            ax1.fill_between(self.wl_em,(Spectra_ct1+Spectra_var1),(Spectra_ct1-Spectra_var1),color="purple",alpha=.25,
                                  label="$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Spectra_ct2+Spectra_var2),(Spectra_ct2-Spectra_var2),color="purple",alpha=.25,
                                  label="$\pm$"+"1"+self.ct_err_name)
                
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.set_title("Corrected Spectra")
            ax1.legend()
           
            
        def unmix_corr_plotter(self,ax1,D1,D2,A1cor,A2cor,label):
    
            
            #central tendency, could be mean or median
            D_ct1=self.ct_function(D1,axis=0)
            D_ct2=self.ct_function(D2,axis=0)
            Acor_ct1=self.ct_function(A1cor,axis=0)
            Acor_ct2=self.ct_function(A2cor,axis=0)
        
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            Acor_var1=self.ct_err_function(A1cor,axis=0)
            Acor_var2=self.ct_err_function(A2cor,axis=0)

            ax1.plot(self.wl_em,D_ct1,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,D_ct2,color="blue",ls='--',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.plot(self.wl_em,Acor_ct1,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[0])
            ax1.plot(self.wl_em,Acor_ct2,color="orange",ls='--',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D_ct1+D_var1),(D_ct1-D_var1),color="blue",alpha=.25,
                                  label=self.fluor_names[0]+" $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(D_ct2+D_var2),(D_ct2-D_var2),color="blue",alpha=.25)
            
            ax1.fill_between(self.wl_em,(Acor_ct1+Acor_var1),(Acor_ct1-Acor_var1),color="orange",alpha=.25,
                                  label=self.fluor_names[1]+" $\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Acor_ct2+Acor_var2),(Acor_ct2-Acor_var2),color="orange",alpha=.25)
            ax1.set_title("FRET Components")
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best')
    
        #End of subroutines
                
        colors=self.clist
        
        if observation=="pixel":
            observation="pix"
        #get observation level dataframe    
        DF=getattr(self,(observation+'_DF')).copy()
        
        for i in tnrange(1):
            print("Getting Figure Sets")
            print("---------------------------------------------------------------------------------")
            #get filtered data (plot_by +specify filtering)
            spec1,labels=self.get_figure_sets(DF,"spectra1",specify,plot_by=plot_by)
            D1dat,labels=self.get_figure_sets(DF,"F_D1",specify,plot_by=plot_by)
            A1dat,labels=self.get_figure_sets(DF,"F_A1",specify,plot_by=plot_by)
            F_AF1,labels=self.get_figure_sets(DF,"F_AF1",specify,plot_by=plot_by)
            A_dir1,labels=self.get_figure_sets(DF,"A_dir1",specify,plot_by=plot_by)
            
            
            spec2,labels=self.get_figure_sets(DF,"spectra2",specify,plot_by=plot_by)
            D2dat,labels=self.get_figure_sets(DF,"F_D2",specify,plot_by=plot_by)
            A2dat,labels=self.get_figure_sets(DF,"F_A2",specify,plot_by=plot_by)
            F_AF2,labels=self.get_figure_sets(DF,"F_AF2",specify,plot_by=plot_by)
            A_dir2,labels=self.get_figure_sets(DF,"A_dir2",specify,plot_by=plot_by)
            
            
            alpha,labels=self.get_figure_sets(DF,"Alpha",specify,plot_by=plot_by)
            
            
        print("Generating Plots")
        print("---------------------------------------------------------------------------------")
        for i in tnrange(len(spec1)):
            N=spec1[i].shape[0]
            Varfig, ((sub1,sub2),(sub3,sub4))=subplots(2,2,figsize=[15,16])
            Varfig.suptitle(labels[i]+' (n='+str(N)+')',fontsize=25)
            
            #unpack variables for computation
            AF1,AF2=F_AF1[i],F_AF2[i]
            Alpha=alpha[i]
            F1=np.vstack(spec1[i])
            F2=np.vstack(spec2[i])
            D1=D1dat[i]
            D2=D2dat[i]
            
            comp_D1=np.tile(self.comps1[0],(len(D1),1))#converts comps array to shape of D1 for simple element wise matrix multiplication
            comp_D2=np.tile(self.comps2[0],(len(D2),1))#converts comps array to shape of D2 for simple element wise matrix multiplication
            comp_A1=np.tile(self.comps1[1],(len(D1),1))#converts comps array to shape of A1 for simple element wise matrix multiplication
            comp_A2=np.tile(self.comps1[1],(len(D1),1))#converts comps array to shape of A2 for simple element wise matrix multiplication
            comp_AF1=np.tile(self.comps1[2],(len(F1),1))#converts comps array to shape of F1 for simple element wise matrix multiplication
            comp_AF2=np.tile(self.comps2[2],(len(F2),1))#converts comps array to shape of F2 for simple element wise matrix multiplication
            
            #perform computations
            D1comp=(D1*comp_D1.T).T#Donor1 vectors
            D2comp=(D2*comp_D2.T).T#Donor2 vectors
            A1_dex=(A_dir1[i]*comp_A1.T).T#A_dir1 vectors
            A2_dex=(A_dir2[i]*comp_A2.T).T#A_dir2 vectors
            A1cor=A1dat[i]-A_dir1[i]#corrected acceptor magnitudes
            A2cor=A2dat[i]-A_dir2[i]#corrected acceptor magnitudes
            A1cor=(A1cor*comp_A1.T).T#corrected acceptor vectors 
            A2cor=(A2cor*comp_A2.T).T#corrected acceptor vectors
            AF1comp=(AF1*comp_AF1.T).T#Autofluorescent 1 vectors
            AF2comp=(AF2*comp_AF2.T).T#Autofluorescent 2 vectors
            F1sub=((F1-AF1comp).T*Alpha).T#Subtract autofluor and correct by alpha[vectors]
            F2sub=(F2-AF2comp)#subtract autofluor[vectors]
            beta=F2sub-F1sub#compute beta vectors
            F1cor=(F1-AF1comp)-A1_dex#subtract direct excitation[vectors]
            F2cor=F2sub-A2_dex#subtract direct excitation[vectors]
            
            Mag1=np.trapz(F1,axis=1)
            Mag2=np.trapz(F2,axis=1)
            F1norm=(F1.T/Mag1).T
            F2norm=(F2.T/Mag2).T
            D1norm=(D1comp.T/Mag1).T
            D2norm=(D2comp.T/Mag2).T
            AF1norm=(AF1comp.T/Mag1).T
            AF2norm=(AF2comp.T/Mag2).T
            alpha_norm=D2norm/D1norm
            beta_norm=(F2norm-AF2norm)-(F1norm-AF1norm)*alpha_norm
            F1cor_norm=(F1cor.T/Mag1).T#normalize spectra by magnitude[vectors]
            F2cor_norm=(F2cor.T/Mag2).T#normalize spectra by magnitude[vectors]
            A1cor_norm=(A1cor.T/Mag1).T
            A2cor_norm=(A2cor.T/Mag2).T
            
            #call plotting functions 
            alpha_plotter(self,sub1,F1norm-AF1norm,F2norm-AF2norm,D1norm,D2norm,labels[i],colors[i%len(colors)])#plot raw spectra variance and donor fit variance
            Beta_plotter(self,sub2,beta_norm,labels[i])#plot beta curve variance: excitation independent
            Spectra_corr_plotter(self,sub3,F1cor_norm,F2cor_norm,labels[i]) #plot corrected spectra variance:excitation dependent
            unmix_corr_plotter(self,sub4,D1norm,D2norm,A1cor_norm,A2cor_norm,labels[i])#unmixed components of the corrected spectra
        
    
    def boxplot(self,param='Eff',ex=2,observation='region',plot_by='group',
                lim_y=[0,1],specify=[],colors=[],rotate_labels=90):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)

        fig, axes = plt.subplots(nrows=1, ncols=1)
        bp=axes.boxplot(data,showfliers=True,widths=.5,notch=False,vert=True,
                        patch_artist=True);

        p=0
        for patch, in zip(bp['boxes']):
            patch.set_facecolor(colors[p])
            patch.set_alpha(.25)
            p=p+1

        axes.set_ylim(lim_y[0],lim_y[1])
        axes.set_ylabel(self.alias[param]+add_label)
        xticks(range(1,len(labels)+1),labels,rotation=rotate_labels);
        
    def significance(self,param='Eff',ex=2,observation='region',plot_by='group',
                threshold=.05,test="T-Test",print_pval=True,specify=[],colors=[]):
        
        if len(colors)==0:
            colors=self.clist
            
        if observation=="pixel":
            observation="pix"
        
        #converts long name to short if long name is used for the parameter
        if param in self.alias_inv:
            param=self.alias_inv[param]
        
        #adds the excitation number to the parameter unless its excitation independent
        if param in self.ex_independent:
            param_ex=param            
            add_label=''
        else:
            param_ex=param+str(ex)
            add_label=' ('+self.ex_names[ex-1]+')'
        
        #pull appropriate dataframe for the observation type
        DF=getattr(self,(observation+'_DF')).copy()
        
        #generate figure sets for plot type
        data,labels=self.get_figure_sets(DF,param_ex,specify,plot_by=plot_by)
        
        stat_options=[f_oneway,mannwhitneyu,ttest_ind,kruskal]#,self.exact_mc_perm_test]
        stat_label=["One Way ANOVA","Mann-Whitney U Test","T-Test","Kruskal-Wallis Test"]#,"Exact_Permutation_Test"]
        
        if test=='All':        
            stat_type=stat_options        
        else:
            stat_type=[stat_options[stat_label.index(test)]]
        
        pvals=np.zeros((len(stat_type),len(data),len(data)))
        
        for t in range(len(stat_type)):
            for i in range(len(data)):
                for j in range(len(data)):
                    pvals[t,i,j]=stat_type[t](data[i],data[j])[1]
                    
        if print_pval==True:
            print(pvals)
            
        for i in range(len(pvals)):
            t_image=(pvals[i]<threshold)
            matshow(t_image,interpolation="none",cmap="Greys",vmax=1,vmin=0)
            xticks(range(len(labels)),labels,rotation=90);
            yticks(range(len(labels)),labels);
            cbar=colorbar(label=stat_label[stat_options.index(stat_type[i])])
            cbar.set_ticks([0,1])
            cbar.set_ticklabels(["p-value>"+str(threshold),"p-value<"+str(threshold)])
            figure()
    
    def plotComps(self,observation='region',
                        specify=[],colors=[],update_values=True):
        def plotter(self,D1,D2,A1,A2,AF1,AF2,update_values):
            #central tendency, could be mean or median
                   
            D1_ct=self.ct_function(D1,axis=0)
            D2_ct=self.ct_function(D2,axis=0)            
            D_var1=self.ct_err_function(D1,axis=0)
            D_var2=self.ct_err_function(D2,axis=0)
            
            A1_ct=self.ct_function(A1,axis=0)
            A2_ct=self.ct_function(A2,axis=0)            
            A_var1=self.ct_err_function(A1,axis=0)
            A_var2=self.ct_err_function(A2,axis=0)
            
            AF1_ct=self.ct_function(AF1,axis=0)
            AF2_ct=self.ct_function(AF2,axis=0)            
            AF_var1=self.ct_err_function(AF1,axis=0)
            AF_var2=self.ct_err_function(AF2,axis=0)
            
            N=D1.shape[0]
            Varfig, (ax1,ax2) =subplots(1,2,figsize=[15,8])
            Varfig.suptitle('Components (n='+str(N)+')',fontsize=25)
            
            ax1.plot(self.wl_em,D1_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,D2_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(D1_ct+D_var1),(D1_ct-D_var1),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(D2_ct+D_var2),(D2_ct-D_var2),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,A1_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[0])
            ax2.plot(self.wl_em,A2_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' '+self.ex_names[1])
            ax1.fill_between(self.wl_em,(A1_ct+A_var1),(A1_ct-A_var1),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(A2_ct+A_var2),(A2_ct-A_var2),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            
            ax1.plot(self.wl_em,AF1_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[0])
            ax2.plot(self.wl_em,AF2_ct,color="green",ls='-',label=self.ct_name+" AutoF "+self.ex_names[1])
            ax1.fill_between(self.wl_em,(AF1_ct+AF_var1),(AF1_ct-AF_var1),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            ax2.fill_between(self.wl_em,(AF2_ct+AF_var2),(AF2_ct-AF_var2),color="green",alpha=.25,label="AutoF $\pm$"+"1"+self.ct_err_name)
            
            ax1.set_title(self.ex_names[0]+' nm')
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Normalized Intensity [Au]')
            ax1.legend(loc='best',fontsize=12)
            
            ax2.set_title(self.ex_names[1]+' nm')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Normalized Intensity [Au]')
            ax2.legend(loc='best',fontsize=12)
            
            show(block=False)
            
            if update_values==True:
                check=str(raw_input('Update which components? (all, none, D, A, or AF) '))

                if check!='none':                
                    self.multicomp=eval(raw_input('Fit using excitation dependent components? (True or False) '))
                    if check=='all' or check=='D':
                        if self.multicomp==False:
                            Donor_freq=int(raw_input('Which excitation should determine donor shape? (1='+self.ex_names[0]+'nm,2='+self.ex_names[1]+'nm) '))
                            if Donor_freq==1:
                                D2_ct=D1_ct
                            if Donor_freq==2:
                                D1_ct=D2_ct
                        self.comps1[0]=D1_ct
                        self.comps2[0]=D2_ct
                    if check=='all' or check=='A':
                        if self.multicomp==False:
                            Acc_freq=int(raw_input('Which excitation should determine accepptor shape? (1='+self.ex_names[0]+'nm,2='+self.ex_names[1]+'nm) '))
                            if Acc_freq==1:
                                A2_ct=A1_ct
                            if Acc_freq==2:
                                A1_ct=A2_ct
                        self.comps1[1]=A1_ct
                        self.comps2[1]=A2_ct
                    if check=='all' or check=='AF':
                        if self.multicomp==False:
                            AF_freq=int(raw_input('Which excitation should determine the autofluorescence shape? (1='+self.ex_names[0]+'nm,2='+self.ex_names[1]+'nm) '))
                            if AF_freq==1:
                                AF2_ct=AF1_ct
                            if AF_freq==2:
                                AF1_ct=AF2_ct
                        self.comps1[2]=AF1_ct
                        self.comps2[2]=AF2_ct
                        
        
        #get observation Dataframe
        if observation=="pixel":
            observation="pix"   
        DF=getattr(self,(observation+'_DF')).copy()
    
        spec1,labels=self.get_figure_sets(DF,"spectra1",specify,plot_by='group')
        spec2,labels=self.get_figure_sets(DF,"spectra2",specify,plot_by='group')
        
        D1=np.vstack(spec1[self.group_index_donor])
        D2=np.vstack(spec2[self.group_index_donor])
        
        A1=np.vstack(spec1[self.group_index_acceptor])
        A2=np.vstack(spec2[self.group_index_acceptor])
        
        if self.no_AutoF==False:
            AF1=np.vstack(spec1[self.group_index_AutoF])
            AF2=np.vstack(spec2[self.group_index_AutoF])
        else:
            AF1=np.zeros(np.shape(D1))
            AF2=np.zeros(np.shape(D2))
        
        D1_Mag=np.trapz(D1,axis=1)
        D2_Mag=np.trapz(D2,axis=1)
        A1_Mag=np.trapz(A1,axis=1)
        A2_Mag=np.trapz(A2,axis=1)
        AF1_Mag=np.trapz(AF1,axis=1)
        AF2_Mag=np.trapz(AF2,axis=1)
        
        D1norm=(D1.T/D1_Mag).T
        D2norm=(D2.T/D2_Mag).T
        A1norm=(A1.T/A1_Mag).T
        A2norm=(A2.T/A2_Mag).T
        AF1norm=(AF1.T/AF1_Mag).T
        AF2norm=(AF2.T/AF2_Mag).T
        
        #call plotting functions 
        plotter(self,D1norm,D2norm,A1norm,A2norm,AF1norm,AF2norm,update_values)
    
    def remCompChannels(self,component='none',ex=1,ind='none'):
        
        if component=='all':
            c=[0,3]
        if component=='Donor':
            c=[0,1]
        if component=='Acceptor':
            c=[1,2]
        if component=='AutoF':
            c=[2,3]
        
        if ex==1:
            if component!='none':           
                self.comps1[c[0]:c[1],ind[0]:ind[1]+1]=0
            figure()
            title('Ex1 Components')
            plot(self.comps1[0],ls='-',c='blue',label='Donor')
            plot(self.comps1[1],ls='-',c='orange',label='Acceptor')
            plot(self.comps1[2],ls='-',c='green',label='AutoF')
            legend()
            
        if ex==2:
            if component!='none':           
                self.comps2[c[0]:c[1],ind[0]:ind[1]+1]=0
            figure()
            title('Ex2 Components')
            plot(self.comps2[0],ls='-',c='blue',label='Donor')
            plot(self.comps2[1],ls='-',c='orange',label='Acceptor')
            plot(self.comps2[2],ls='-',c='green',label='AutoF')
            legend()
            
    def plotGamma(self,observation='region',specify=[],colors=[],
                  spec_range=(0,1),g_range=(0,.5),ratio_range=(0,2),update_values=True):
        def plotter(self,D1spec,D2spec,A1spec,A2spec,D1,D2,A1,A2,spec_range,g_range,ratio_range,update_values):
            #central tendency, could be mean or median
            Dratio_spec=D2spec/D1spec
            Dr=D2/D1
            Dr_fit=self.ct_function(Dr)
            
            Aratio_spec=A1spec/A2spec
            Ar=A1/A2
            Ar_fit=self.ct_function(Ar)            
            
            Dr_ct=self.ct_function(Dratio_spec,axis=0)
            Dr_var=self.ct_err_function(Dratio_spec,axis=0)
            Ar_ct=self.ct_function(Aratio_spec,axis=0)
            Ar_var=self.ct_err_function(Aratio_spec,axis=0)
            
            gamma=np.outer(Dr,Ar).flatten()
            
            print('D2/D1: '+str(Dr_fit))
            print('A1/A2: '+str(Ar_fit))
            print('Gamma: '+str(Dr_fit*Ar_fit))
            specfig, (ax1) =subplots(1,1,figsize=[15,8])
            
            ax1.plot(self.wl_em,Dr_ct,color="blue",ls='-',label=self.ct_name+" "+self.fluor_names[0]+' ratio ('+self.ex_names[1]+'/'+self.ex_names[0]+')')
            ax1.fill_between(self.wl_em,(Dr_ct+Dr_var),(Dr_ct-Dr_var),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Dr_ct+Dr_var),(Dr_ct-Dr_var),color="blue",alpha=.25,label=self.fluor_names[0]+"$\pm$"+"1"+self.ct_err_name)
            ax1.axhline(Dr_fit,color='blue',ls='--',label=self.fluor_names[0]+' ratio fit')
            
            ax1.plot(self.wl_em,Ar_ct,color="orange",ls='-',label=self.ct_name+" "+self.fluor_names[1]+' ratio ('+self.ex_names[0]+'/'+self.ex_names[1]+')')
            ax1.fill_between(self.wl_em,(Ar_ct+Ar_var),(Ar_ct-Ar_var),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            ax1.fill_between(self.wl_em,(Ar_ct+Ar_var),(Ar_ct-Ar_var),color="orange",alpha=.25,label=self.fluor_names[1]+"$\pm$"+"1"+self.ct_err_name)
            ax1.axhline(Ar_fit,color='orange',ls='--',label=self.fluor_names[1]+' ratio fit')
            
            ax1.set_ylim(spec_range[0],spec_range[1])
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Ratio [Au]')
            ax1.legend(loc=1,fontsize=12)
            
            axt=ax1.twinx()

            axt.plot(self.wl_em,self.comps1[0],color="blue",ls=':',label=self.fluor_names[0]+' ref')
            
            axt.plot(self.wl_em,self.comps1[1],color="orange",ls=':',label=self.fluor_names[1]+' ref')

            axt.set_xlabel('Wavelength [nm]')
            axt.set_ylabel('Intensity [Au]')
            axt.legend(loc=2,fontsize=12)
            
            histfig, (ax2,ax3) =subplots(1,2,figsize=[15,8])

            ax2.hist(Dr,normed=True,color='blue',histtype='stepfilled',bins=50,range=ratio_range,alpha=.5,label=self.fluor_names[0]+' ratio ('+self.ex_names[1]+'/'+self.ex_names[0]+')')
            ax2.axvline(Dr_fit,color='blue',ls='--')
            ax2.hist(Ar,normed=True,color='orange',histtype='stepfilled',bins=50,range=ratio_range,alpha=.5,label=self.fluor_names[1]+' ratio ('+self.ex_names[0]+'/'+self.ex_names[1]+')')
            ax2.axvline(Ar_fit,color='orange',ls='--')
            
            ax3.hist(gamma,normed=True,color='green',histtype='stepfilled',bins=50,range=g_range,alpha=.5,label=self.fluor_names[1]+' ratio ('+self.ex_names[0]+'/'+self.ex_names[1]+')')
            ax3.axvline(Dr_fit*Ar_fit,color='green',ls='--')
            ax2.set_ylabel('Frequency')
            ax2.set_xlabel('Ratio [Au]')
            ax2.legend(loc='best',fontsize=12)
            
            ax3.set_ylabel('Frequency')
            ax3.set_xlabel('Gamma')
            
            show(block=False)
            
            if update_values==True:
                check=eval(raw_input('Are you sure you want to update the Gamma parameter? (True or False) '))

                if check==True:                
                    self.gamma=Dr_fit*Ar_fit
                        
        
        #get observation Dataframe
        if observation=="pixel":
            observation="pix"   
        DF=getattr(self,(observation+'_DF')).copy()
    
        spec1,labels=np.array(self.get_figure_sets(DF,"spectra1",specify,plot_by='group'))
        spec2,labels=np.array(self.get_figure_sets(DF,"spectra2",specify,plot_by='group'))
        
        D_D1dat,labels=self.get_figure_sets(DF,"D_D1",specify,plot_by='group')
        D_D2dat,labels=self.get_figure_sets(DF,"D_D2",specify,plot_by='group')
        A_A1dat,labels=self.get_figure_sets(DF,"A_A1",specify,plot_by='group')
        A_A2dat,labels=self.get_figure_sets(DF,"A_A2",specify,plot_by='group')
        
        D_AF1dat,labels=self.get_figure_sets(DF,"D_AF1",specify,plot_by='group')
        D_AF2dat,labels=self.get_figure_sets(DF,"D_AF2",specify,plot_by='group')
        A_AF1dat,labels=self.get_figure_sets(DF,"A_AF1",specify,plot_by='group')
        A_AF2dat,labels=self.get_figure_sets(DF,"A_AF2",specify,plot_by='group')
        
        D_AF1,D_AF2=D_AF1dat[self.group_index_donor],D_AF2dat[self.group_index_donor]
        A_AF1,A_AF2=A_AF1dat[self.group_index_acceptor],A_AF2dat[self.group_index_acceptor]
        
        comp_D_AF1=np.tile(self.comps1[2],(len(D_AF1),1))
        comp_D_AF2=np.tile(self.comps2[2],(len(D_AF2),1))        
        comp_A_AF1=np.tile(self.comps1[2],(len(A_AF1),1))
        comp_A_AF2=np.tile(self.comps2[2],(len(A_AF2),1))
    
        D1spec=np.vstack(spec1[self.group_index_donor])-(D_AF1*comp_D_AF1.T).T
        D2spec=np.vstack(spec2[self.group_index_donor])-(D_AF2*comp_D_AF2.T).T
        A1spec=np.vstack(spec1[self.group_index_acceptor])-(A_AF1*comp_A_AF1.T).T
        A2spec=np.vstack(spec2[self.group_index_acceptor])-(A_AF2*comp_A_AF2.T).T
        
        D1,D2=D_D1dat[self.group_index_donor],D_D2dat[self.group_index_donor]
        A1,A2=A_A1dat[self.group_index_acceptor],A_A2dat[self.group_index_acceptor]
        
        #call plotting functions 
        plotter(self,D1spec,D2spec,A1spec,A2spec,D1,D2,A1,A2,spec_range,g_range,ratio_range,update_values)
                        
    def plotExRatios(self,observation='region',ratio_range=(0,100),bins=100,
                        specify=[],colors=[],update_values=True):
        
        if self.no_FRETStandard==True:
            print('No FRET Standard to calculate excitation ratios from')
            return
            
        #get observation Dataframe
        if observation=="pixel":
            observation="pix"   
        DF=getattr(self,(observation+'_DF')).copy()
    
        exRatio1,labels=np.array(self.get_figure_sets(DF,"eD_eA1",specify,plot_by='group'))
        exRatio1=exRatio1[np.where(labels=='FRET_Standard')][0]
        exRatio2,labels=np.array(self.get_figure_sets(DF,"eD_eA2",specify,plot_by='group'))
        exRatio2=exRatio2[np.where(labels=='FRET_Standard')][0]
        N1=exRatio1.shape[0]
        N2=exRatio2.shape[0]
        exRatio1_ct=self.ct_function(exRatio1,axis=0)        
        exRatio2_ct=self.ct_function(exRatio2,axis=0)
        
        histfig, ax =subplots(1,1,figsize=[15,8])

        ax.hist(exRatio1,normed=True,color='red',histtype='stepfilled',bins=bins,
                range=ratio_range,alpha=.5,label=self.ex_names[0]+' nm (N='+str(N1)+')')
        ax.axvline(exRatio1_ct,color='red',ls='--')
        ax.hist(exRatio2,normed=True,color='green',histtype='stepfilled',bins=bins,
                range=ratio_range,alpha=.5,label=self.ex_names[1]+' nm (N='+str(N2)+')')
        ax.axvline(exRatio2_ct,color='green',ls='--')
        
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Excitation Ratio (eD/eA)')
        ax.legend(loc='best',fontsize=12)
        
        show(block=False)
        
        print('Excitation Ratio ('+str(self.ex_names[0])+' nm):'+str(exRatio1_ct))
        print('Excitation Ratio ('+str(self.ex_names[1])+' nm):'+str(exRatio2_ct))
        if update_values==True:
            check=eval(raw_input('Are you sure you want to update the excitation ratio parameter? (True or False) '))
            if check==True:
                self.exRatio1=exRatio1_ct
                self.exRatio2=exRatio2_ct

    def create_calibration_DF(self,to_xlsx=True):
        """ Make calibration DataFrame to read in
        info needed
        """
        self.calib_DF=[]
        d={'Calibration_Name':self.CalibName}
        df=pd.DataFrame.from_dict(d, orient='index').T
    
        df['comps1']=str(self.comps1.tolist())
        df['comps2']=str(self.comps2.tolist())
        df['wl_em']=str(self.wl_em.tolist())
        df['Ex1']=self.ex_names[0]
        df['Ex2']=self.ex_names[1]
        df['D_name']=self.fluor_names[0]
        df['A_name']=self.fluor_names[1]
        df['Qd']=self.Qd
        df['Qa']=self.Qa
        df['gamma']=self.gamma
        df['FS_name']=self.FS_name
        df['exRatio1']=self.exRatio1
        df['exRatio2']=self.exRatio2
        df['Date']=self.date
           
        self.calib_DF=df.T
        
        if to_xlsx==True:
            check=eval(raw_input('Write .xlsx calibration file? (True or False) '))
            if check==True:
                print('Exporting')
                writer = pd.ExcelWriter('CalibrationFiles/Calibration_xlsx/'+self.CalibName+'.xlsx')                
                self.calib_DF.to_excel(writer,'Info')
        
        return (self.calib_DF)