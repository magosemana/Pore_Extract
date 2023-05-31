# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:42:11 2023

@author: Joao
"""
#This script will use SUDODEM to execute the post-treatment of the pnExtract data
import math
import numpy as np
import pandas as pd
import os
import sys
from sudodem import utils # module utils has some auxiliary functions
from sudodem import _superquadrics_utils
from sudodem._superquadrics_utils import *
#from pathlib import Path

pathScript=str(sys.argv[0])
chk=pathScript.rfind('/')
if chk>0:
    pathScript=pathScript[0:chk]
else:
    pathScript=os.getcwd()+'/'

sys.path.append(pathScript)
print('\n')
print('Execution path : '+ pathScript)
print('')

from functions  import *
from functionsJ import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#################################       Variables      ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

voxel_fnm="p_s_5cm_loose_cnr_ellipsoid"     #voxel file name
voxel_cut=[0.00346102,0.0347862,0.0337112]  #relative position of pnExtract results
voxel=2.083333e-05                          #dimension of a voxel
voxel_multiplier=5                          #multiply voxel dimension (for testing)
pathFile=""                                 #path to voxel files (if needed)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###############################       Load pnExtract      ###############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if any of the strings is empty ask the user for new place
if not pathFile:
     pathFile=pathScript
#prepare path to save results, add folder if it does not exist
svPath=os.path.join(pathFile,"Pore_Results")
if not os.path.exists(svPath):
        os.mkdir(svPath)
#check if pnExtract summary files exists
t_name=os.path.join(svPath,"throat_to_DEM.dat")
p_name=os.path.join(svPath,"pore_to_DEM.dat")
if os.path.exists(t_name) and os.path.exists(p_name):  
    print("Pore and throat summary files found.")
    t_data=pd.read_csv(t_name)
    p_data=pd.read_csv(p_name)
else:
    print("Pore and throat summary files not found, loading PnExtract data.")
    #check files existence, quit if file was not found
    if checkFiles(pathFile,voxel_fnm):
        O.exitNoBacktrace()
    ########## Input data from pn_extract #######################
    #load pore data (_node2.dat and _node.dat files)
    data_pore=GetPoreData(os.path.join(pathFile,voxel_fnm))
    #load throath data (_link2.dat and _link1.dat files)
    data_throat=GetThroatData(os.path.join(pathFile,voxel_fnm))
    #delete pores in the 0.1 part extremes of the specimen
    [p_data,data_throat_noborder]=DeleteBorderElements(data_pore,data_throat,0.1)
    t_data=data_throat_noborder[data_throat_noborder["Throat_Radius"]>9.6*0.000001]  #delete all the small fine particles that are artifical (less than Do/6.5)
    #Check if throath normal exists
    if os.path.exists(os.path.join(pathFile,voxel_fnm+'_SURF.dat')):
        #calculate throat normals and add to the file
        throat_normal=GetThroatInfo(os.path.join(pathFile,voxel_fnm),0.1)
        #rename t_id column to append be able to join table to t_data
        throat_normal.rename(columns = {'t_id':'Throat_index'}, inplace = True)
        t_data=t_data.merge(throat_normal)
    else:
        print("File "+voxel_fnm+"_SURF.dat was not found. Constriction normals will be attributed as nul vectors.")
        #if file do not exist create zero vector in the following properties : "norm_vector_x","norm_vector_y","norm_vector_z"
        z=np.zeros([len(t_data)])
        t_data["norm_vector_x"]=z
        t_data["norm_vector_y"]=z
        t_data["norm_vector_z"]=z

    #update pore and throat data using the voxel_cut data
    t_data['t_x']=t_data.t_x.values+voxel_cut[0]
    t_data['t_y']=t_data.t_y.values+voxel_cut[1]
    t_data['t_z']=t_data.t_z.values+voxel_cut[2]
    p_data['x']=p_data.x.values+voxel_cut[0]
    p_data['y']=p_data.y.values+voxel_cut[1]
    p_data['z']=p_data.z.values+voxel_cut[2]
    #save pore data summary files as csv  (coma separated values)
    t_data.to_csv(t_name,index=False)
    p_data.to_csv(p_name,index=False)
    #delete tables(DataFrame) that are no longer needed
    del data_throat_noborder, data_pore, data_throat, throat_normal

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################       DEM      #####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Prepare DEM data")
### DEM BASE DATA
#Mechanical parametrs
normal_stiffness=5.65e5                
tangent_stiffness=5.65e5    
friction_angle=math.atan(0.00)  #input
density=26.5e6 # to increase the time step *e4 for fine particles
mat_superellipse= SuperquadricsMat(label="mat1",Kn=normal_stiffness,Ks=tangent_stiffness,betan=0.61,betas=0.61,frictionAngle=friction_angle,density=density)   
O.materials.append(mat_superellipse)

###### Load filter particles ######

    ## Load data
axis_list= np.load(os.path.join(pathFile,'axis_list.npy'))
angle_list= np.load(os.path.join(pathFile,'angle_list.npy'))
position_list= np.load(os.path.join(pathFile,'position_list.npy'))
rx_list=np.load(os.path.join(pathFile,'rx_list.npy'))
ry_list=np.load(os.path.join(pathFile,'ry_list.npy'))
rz_list=np.load(os.path.join(pathFile,'rz_list.npy'))

    ## add particles
for n in range(len(position_list)):
    body = NewSuperquadrics2(rx_list[n],ry_list[n],rz_list[n],1,1,mat_superellipse,False,False)
    body.state.pos=position_list[n]
    body.state.ori=(axis_list[n],angle_list[n])
    O.bodies.append(body)
    body.shape.color=(0.5,0.6,0.7)
    body.dynamic = False
    #body.shape.wire=True

    ## count particles
nb_filter=len(O.bodies)
print "Number of filter particles",nb_filter

###### Create constrictions as particles ######
#Create speheres where constrictions should be.
vx =voxel*voxel_multiplier
for t_row in t_data.itertuples():
    body = NewSuperquadrics2(t_row.Throat_Radius+vx, t_row.Throat_Radius+vx, t_row.Throat_Radius+vx, 1, 1, mat_superellipse, False, True)
    body.state.pos=[t_row.t_x, t_row.t_y, t_row.t_z]
    O.bodies.append(body)
    body.shape.color=(0.0,0.0,01)
    body.dynamic = False
    #body.shape.wire=False

    #count constrictions
nb_cons=len(O.bodies)-nb_filter
print "Number of constrictions", nb_cons

###### Create pores as particles ######
#Create speheres where pores should be.
for p_row in p_data.itertuples():
    body = NewSuperquadrics2(p_row.Pore_radius+vx, p_row.Pore_radius+vx, p_row.Pore_radius+vx, 1, 1, mat_superellipse, False, True)
    body.state.pos=[p_row.x, p_row.y, p_row.z]
    O.bodies.append(body)
    body.shape.color=(0.0,0.0,01)
    body.dynamic = False
    #body.shape.wire=False

    #count pore
nb_pore=len(O.bodies)-nb_cons-nb_filter
print "Number of pores", nb_pore  

###### Prepare DEM to check for contacts ######
newton=NewtonIntegrator(damping = 0,gravity=(0.,9.81,0),label="newton",isSuperquadrics=1) # isSuperquadrics: 1 for superquadrics

O.engines=[
   ForceResetter(), InsertionSortCollider([Bo1_Superquadrics_Aabb(),Bo1_Wall_Aabb()],verletDist=0.0001),
   InteractionLoop(
      [Ig2_Wall_Superquadrics_SuperquadricsGeom(),    Ig2_Superquadrics_Superquadrics_SuperquadricsGeom()],
      [Ip2_SuperquadricsMat_SuperquadricsMat_SuperquadricsPhys()], # collision "physics"
      [SuperquadricsLaw()]   # contact law
   ),
   newton,
]

O.dt=0.000000001
O.run(1,True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####################################       Run      ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#create contriction objects
constr=constrictionCheck(O, t_data, nb_filter, nb_cons, svPath)
#create pore objects
pore=poreCheck(O, constr, p_data, nb_filter, nb_cons, svPath)
#plot pore values
poreResultsPlot(pore)

#Quit SUDODEM
O.exitNoBacktrace()