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
pathScript=pathScript[0:(pathScript.rfind('/'))]
sys.path.append(pathScript)
print('\n')
print('Execution path : '+ pathScript)
print('')

from functions  import *
from functionsJ import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###################################       Variables      ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
voxel_fnm="p_s_vox_20_loose_cnr_ellipsoid"
pathFile=""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#################################       Load pnExtract      ##################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if any of the strings is empty ask the user for new place
if not pathFile:
     pathFile=pathScript
#prepare path to save results
svPath=os.path.join(pathFile,"Pore_Results")
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
    #calculate throat normals and add to the file
    throat_normal=GetThroatInfo(os.path.join(pathFile,voxel_fnm),0.1)
    throat_normal.rename(columns = {'t_id':'Throat_index'}, inplace = True)
    t_data=t_data.merge(throat_normal)
    #save pore data summary files as csv  (coma separated values)
    t_data.to_csv(t_name,index=False)
    p_data.to_csv(p_name,index=False)
    #delete data that will not be needed
    del data_throat_noborder, data_pore, data_throat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#######################################       DEM      #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Prepare DEM data")
### DEM BASE DATA
#Mechanical parametrs
normal_stiffness=5.65e5                
tangent_stiffness=5.65e5    
friction_angle=math.atan(0.00)  #input
density=26500000 # to increase the time step *e4 for fine particles
mat_superellipse= SuperquadricsMat(label="mat1",Kn=normal_stiffness,Ks=tangent_stiffness,betan=0.61,betas=0.61,frictionAngle=friction_angle,density=density)   
O.materials.append(mat_superellipse)

#Dimensions of the box
lintmax=0.020
bo_de=lintmax/2
width=lintmax*5
depth=lintmax*5
height=lintmax*5

### Load filter ###

        ##Load particles
axis_list= np.load(os.path.join(path,'axis_list.npy'))
angle_list= np.load(os.path.join(path,'angle_list.npy'))
position_list= np.load(os.path.join(path,'position_list.npy'))
rx_list=np.load(os.path.join(path,'rx_list.npy'))
ry_list=np.load(os.path.join(path,'ry_list.npy'))
rz_list=np.load(os.path.join(path,'rz_list.npy'))
zero_list=[]
one_list=[]

### ALI'S WEIRD WALL DELTION
number_of_particles=0
n=0
for xyz in position_list:
    body = NewSuperquadrics2(rx_list[n],ry_list[n],rz_list[n],1,1,mat_superellipse,False,False)
    body.state.pos=position_list[n]
    body.state.ori=(axis_list[n],angle_list[n])
    O.bodies.append(body)
        #body.shape.wire=True
    body.shape.color=(0.5,0.6,0.7)
        #body.dynamic = False
    number_of_particles=number_of_particles+1  
    n=n+1
print "Number of particles", number_of_particles 

total_bodies=len(O.bodies)
print "Total_bodies",total_bodies

### CONSTRICTION DATA
c_nb=0
n=0
c_radius=t_data.Throat_Radius.values
c_x=t_data.t_x.values
c_y=t_data.t_y.values
c_z=t_data.t_z.values

        #Create speheres where constrictions should be.
voxel=6.250000e-05
ratio_i=voxel
for xyz in c_radius:
    body = NewSuperquadrics2(c_radius[n]+ratio_i,c_radius[n]+ratio_i,c_radius[n]+ratio_i,1,1,mat_superellipse,False,True)
    body.state.pos=[c_x[n]+0.0396102,c_y[n]+0.0397862,c_z[n]+0.0387112]
    O.bodies.append(body)
    body.shape.wire=False
    body.shape.color=(0.0,0.0,01)
    body.dynamic = False
    c_nb=c_nb+1  
    n=n+1
print "Number of constrictions", c_nb     

### ENGINE AND RUN
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#######################################       Run      #######################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
O.dt=0.000000001
O.run(1,True)
for i in O.bodies:
        i.dynamic = False 

constr=constrictionCheck(O, t_data, total_bodies, svPath)
pore=poreCheck(O, constr, p_data, svPath)
porePlot(pore)

#Quit SUDODEM
O.exitNoBacktrace()