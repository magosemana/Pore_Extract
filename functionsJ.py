#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:42:11 2023

@author: Joao
"""
import numpy as np
import pandas as pd
import dill
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#####################################       Classes      #####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Constriction:
    def __init__(self, ID, pos, rad, pr1, pr2, ctcPt, grID, norm):
        self.id=ID
        self.position = pos
        self.radius = rad
        self.pore1 = pr1
        self.pore2 = pr2
        self.contactPoint = ctcPt
        self.grainID = grID
        self.normal=norm

class Pore:
    def __init__(self, ID, pos, rVol, rad, cnstr, pts, gr ,cHull, shpf, edim, cd1, cd2):
        self.id=ID
        self.position = pos
        self.real_volume = rVol
        self.radius = rad
        self.constrictions = cnstr
        self.points = pts
        self.grains = gr
        self.triang = cHull.simplices
        self.hull_volume =cHull.volume
        self.shape_factor= shpf
        self.equivalent_diameter= edim
        self.coord_old= cd1
        self.coord_new= cd2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################       Functions      ####################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def grCtcList(O,total_bodies,path):
    #Return the contact lists between "real grains"
    ctcL=[]
    for c in O.interactions:
        if c.id1<total_bodies and c.id1<total_bodies:
            ct=c.geom.contactPoint
    ctcL.append([c.id1,c.id2,ct[0],ct[1],ct[2]])

    c_val=["id1","id2","x","y","z"]
    df=pd.DataFrame(data=ctcL,columns=c_val)
    df.to_csv(os.path.join(path,"grain_contact_list"))

    return ctcL

def constrictionCheck(O, t_data, total_bodies, path):
    print("Prepare constriction object")
    c_id_good=[]
    c_id_bad=[]
    constr=[]
    
    c_id=t_data.Throat_index.values
    c_pore1=t_data.Pore_1_index.values
    c_pore2=t_data.Pore_2_index.values
    c_radius=t_data.Throat_Radius.values
    c_normal=np.transpose(np.array([t_data.norm_vector_x.values,t_data.norm_vector_y.values,t_data.norm_vector_z.values]))
    k=0
    #For each constriction, check the number of contacts with grains and classify it as good or bad constriction
    for i in O.bodies:
        #constrictions have id's higher than the number of grains
        if (i.id>=total_bodies):
            non=0
            ctc = np.zeros((10,3))
            grID= np.zeros((10,1))
            #for each contact of the constriction
            for ct in O.bodies[i.id].intrs():
                if ct.geom.PenetrationDepth>0:
                        #print "pebetration",k.geom.PenetrationDepth 
                        if ct.id1==i.id:
                            ID=ct.id2
                        else:
                            ID=ct.id1
                        
                        if ID<=total_bodies:
                            ctc[non,:]=ct.geom.contactPoint
                            grID[non]=ID
                            non=non+1

            if non>=3:
                #print "contact with 3 filter particles"
                c_id_good.append(c_id[k])
                O.bodies[i.id].shape.color=(1,0,0)
                cns=Constriction(c_id[k], O.bodies[i.id].state.pos, c_radius[k], c_pore1[k], c_pore2[k], ctc[0:non], grID[0:non], c_normal[k])
                constr.append(cns)
            else:
                #print "contact with 3 filter particles"
                c_id_bad.append(c_id[k])
                O.bodies[i.id].shape.color=(0,1,0)

            k=k+1

    print "number_good",len(c_id_good)
    print "number_bad",len(c_id_bad)
    with open(os.path.join(path,"constr_obj.pkl"), 'wb') as f:
        dill.dump(constr, f)

        return constr

def poreCheck(O,constr,p_data, path):
    print("Prepare pore object")
    pore=[]

    #Make a list of pores connected by each good constriction
    pore1=[o.pore1 for o in constr]
    pore2=[o.pore2 for o in constr]

        #Load pore data 
    p_x=p_data.x.values
    p_y=p_data.y.values
    p_z=p_data.z.values
    p_id=p_data.Pore_index.values
    p_volume=p_data.Pore_volume.values
    p_rad=p_data.Pore_radius.values
    p_shape=p_data.Pore_Shape_factor.values
    p_edim=p_data.Pore_Equivalent_Diameter.values
    p_coord=p_data.Pore_Coordination_number.values
    #Create Lambda function that searches for x==y and returns the index of the value
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        #len calculates the size of vector xs
        #range create is 1:10 in matlab
        #zip creates iteration object where the first column is the vector xs and the second column is the range.
        #the for matches y to the first column of the iteration object and i to the second. If x==y at each iteration, then returns the value i

    #Create pore objects
    for p in range(len(p_rad)):
        cdNb=0 #nb de coordination
        chk=get_indexes(p_id[p],pore1)
        chk2=get_indexes(p_id[p],pore2)
        
        if not chk and not chk2:
            #if both list are empty go next pore
            continue
        elif (len(chk)+len(chk2))<2:
            #if less than 2 constrictions per pore, go next
            continue

        chk=chk+chk2

        #for each of the constrictions connected to the pore p, read characteristics
        
        for j in chk:
            cdNb+=1                               #coordination number
            if j==chk[0]:
                ctcPt=constr[j].contactPoint    #contact points
                grID=constr[j].grainID          #grains in contact ID
                cID=np.array([constr[j].id])    #constriction ID
            else:
                ctcPt = np.concatenate((ctcPt,constr[j].contactPoint)) 
                grID= np.concatenate((grID,constr[j].grainID))
                cID= np.concatenate((cID,[constr[j].id]))

        #Transform grID into array and get grains conected to constrictions
        grID=np.sort(np.unique(np.array(grID))) 
        
        #find the contact points between the grains
        ctcGr=[]
        for j in grID:
            # grID is no longer int for some reason
            J=int(j)
            #for each contact of grain j
            for k in O.bodies[J].intrs():
                #if contact has already been checked, jump it
                if J>k.id1 or J>k.id2 or k.geom.PenetrationDepth<=0:
                    continue

                #check if the grains ID's of the contact belong to the grs ids of the constrictions
                bool_array = np.in1d(grID, [k.id1, k.id2])
                gr=grID[bool_array];
                #if both grains belong to it, add the contact point into the matrix
                if np.size(gr)>1:
                    ctc=k.geom.contactPoint
                    print ctc
                    ctcGr.append([ctc[0],ctc[1],ctc[2]])

        #add grain contact points to pore contacts
        ctcPt = np.array(ctcPt) #transform in array
        ctcGr = np.array(ctcGr)
        ctcPt = np.concatenate((ctcPt,ctcGr)) 
        #calculate convex hull
        cHull= ConvexHull(ctcPt)

        #create pore
        pr=Pore(p_id[p], [p_x[p],p_y[p],p_z[p]], p_volume[p], p_rad[p], cID, ctcPt, grID, cHull, p_shape[p], p_edim[p], p_coord[p], cdNb)
        #create vtk file
        vtkSurfPlot(cHull, p_id[p], path)
        pore.append(pr)

    with open(os.path.join(path,"pore_obj.pkl"), 'wb') as f:
        dill.dump(pore, f)

    return pore

def porePlot(pore):
    #get the data from pore object to be plot
    ratZ=np.zeros((len(pore),1))
    ratV=np.zeros((len(pore),1))
    r=range(len(pore))
    for p in r:
        ratV[p]= pore[p].real_volume/pore[p].hull_volume
        ratZ[p]=pore[p].coord_old/pore[p].coord_new

    plt.figure(1)
    plt.plot(r,ratV,'ro')
    plt.axhline(y = 1, color = 'k', linestyle = '-')
    plt.xlabel('ID')
    plt.ylabel('V ratio')
    plt.title('Pore V ratio')

    plt.figure(2)
    plt.plot(r,ratZ,'ro')
    plt.axhline(y = 1, color = 'k', linestyle = '-')
    plt.xlabel('ID')
    plt.ylabel('Z ratio')
    plt.title('Pore Z ratio')
    plt.show()

def vtkSurfPlot(cHull, poreID, path):
    #prepare data to vtkplot
    nbPt=cHull.points.shape[0] #nb of points
    nbTri=cHull.simplices.shape[0] #nb of triangles
    
    #check if folder exists
    pth=os.path.join(path,"poreVtk")
    if not os.path.exists(pth):
        os.mkdir(pth)
    #Create or overwrite file ('w' option)
    fnm=os.path.join(pth,"Pore"+str(poreID)+".vtk")
    f = open(fnm, 'w'); 
    # VTK files contain five major parts
    # 1. VTK DataFile Version
    f.write('# vtk DataFile Version 2.0\n');
    # 2. Title
    f.write('Pore VTK draw from Python\n');
    # 3. The format of data
    f.write('ASCII\n');
    # 4. Type of Dataset
    f.write('DATASET POLYDATA\n');
    # 5. Describe the data set
    #   Save points
    f.write('POINTS {:d} float\n'.format(nbPt));
    for i in range(nbPt):
        k=cHull.points[i]
        f.write("{:.5f} {:.5f} {:.5f} \n".format(k[0],k[1],k[2]));

    #   Save the triangles
    f.write("\nPOLYGONS {:d} {:d}\n".format(nbTri,4*nbTri));
    for i in range(nbTri):
        k=cHull.simplices[i]
        f.write("3 {:d} {:d} {:d}\n".format(k[0],k[1],k[2]))


def checkFiles(path,voxel_fnm):
    val=False
    if not os.path.exists(os.path.join(path,voxel_fnm+'_node2.dat')):
        print("File "+voxel_fnm+"_node2.dat was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,voxel_fnm+'_node1.dat')):
        print("File "+voxel_fnm+"_node1.dat was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,voxel_fnm+'_link2.dat')):
        print("File "+voxel_fnm+"_link2.dat was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,voxel_fnm+'_link1.dat')):
        print("File "+voxel_fnm+"_link1.dat was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,voxel_fnm+'_SURF.dat')):
        print("File "+voxel_fnm+"_SURF.dat was not found.")
        val=True
    
    if val:
        print("Check if pnExtract was executed beforehand.")

    return val
    
    