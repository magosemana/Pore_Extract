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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##################################       Classes      ###################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Constriction:
    def __init__(self, t_data, ctcPt, grID,):
        self.id=t_data.Throat_index                                 #constr ID
        self.position = np.array([t_data.t_x,\
         t_data.t_y, t_data.t_z])                                   #constr center
        self.radius = t_data.Throat_Radius                          #constr radius
        self.pore1 = t_data.Pore_1_index                            #ID pore 2
        self.pore2 = t_data.Pore_2_index                            #ID pore 1
        self.contactPoint = ctcPt                                   #contact pts
        self.grainID = grID                                         #ID grains in contact
        self.normal=np.array([t_data.norm_vector_x,\
         t_data.norm_vector_y, t_data.norm_vector_z])               #constr normal
class Pore:
    def __init__(self, p_data, cnstr, pts, gr, cHull, cdNb):
        self.id=p_data.Pore_index                                   #pore ID
        self.position = [p_data.x, p_data.y, p_data.z]              #pore center
        self.radius=p_data.Pore_radius                              #pore inscribed rad
        self.constrictions = cnstr                                  #pore's constr ID
        self.points = pts                                           #points forming pore
        self.grains = gr                                            #pore's grain ID
        self.volume_real = p_data.Pore_volume                       #pnExtract volume
        self.volume_hull = cHull.volume                             #convex hull volume
        self.volume_ratio= p_data.Pore_volume/cHull.volume          #volume ratio
        self.triang = cHull.simplices                               #convex hull surf
        self.shape_factor= p_data.Pore_Shape_factor                 #Edgar's shape fact
        self.equivalent_diameter= p_data.Pore_Equivalent_Diameter   #Edgar's equiv dim
        self.coord_old= p_data.Pore_Coordination_number             #pnExtract coord nb
        self.coord_new= cdNb                                        #new coord nb
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#################################       Functions      #################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Create objects
def constrictionCheck(O, t_data, nb_filter, nb_cons, path):
    print("Prepare constriction object")
    #initialize variables
    c_id_good=[]            #list of good pore IDs
    c_id_bad=[]             #list of bad pore IDs
    constr=[]               #list of constriction objects
    
    #For each constriction, check the number of contacts with grains and classify it as good or bad constriction. Constriction are bodies created after the grains.
    i=0
    for t_row in t_data.itertuples():
        #constrictions have id's higher than the number of grains
        nbIt=0
        b_id=i+nb_filter #body ID
        ctc = []
        grID= []
        #for each interaction of the constriction t_row
        for ct in O.bodies[b_id].intrs():
            #check if there is contact and if one of the id's of the contact is a filter grain
            if ct.geom.PenetrationDepth>0 and (ct.id1<=nb_filter or ct.id2<=nb_filter):
                pt=ct.geom.contactPoint #transform from vector3 to normal list
                ctc.append([pt[0],pt[1],pt[2]])
                grID.append(min(ct.id1, ct.id2))
                nbIt=nbIt+1

        if nbIt>=3:
            #print "contact with 3 filter particles"
            c_id_good.append(t_row.Throat_index)
            O.bodies[b_id].shape.color=(1,0,0)
            cns=Constriction(t_row, ctc, grID)
            constr.append(cns)
        else:
            #print "contact with 3 filter particles"
            c_id_bad.append(t_row.Throat_index)
            O.bodies[b_id].shape.color=(0,1,0)

        i+=1


    print "Number of good constrictions :",len(c_id_good)
    print "Number of bad constrictions :",len(c_id_bad)

    #Create a contriction vtk file
    vtkSpherePlot(constr, path, 'Constrictions.vtk')
    #save constrictions objects in a file
    with open(os.path.join(path,"constr_obj.pkl"), 'wb') as f:
        dill.dump(constr, f)

        return constr
def poreCheck(O, constr, p_data, nb_filter, nb_cons, path):
    print("Prepare pore object")
    poreList=[]

    #Make a list of pores connected by each good constriction
    pore1=[o.pore1 for o in constr]
    pore2=[o.pore2 for o in constr]

    #Create Lambda function that searches for x==y and returns the index of the value
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        #len calculates the size of vector xs
        #range create is 1:10 in matlab
        #zip creates iteration object where the first column is the vector xs and the second column is the range.
        #the for matches y to the first column of the iteration object and i to the second. If x==y at each iteration, then returns the value i

    #Create pore objects
    i=0
    for p_row in p_data.itertuples():
        cdNb=0 #nb de coordination
        ### Check constrictions : only pores with at least 2 constrictions are saved
            #Identify constrictions containing pore p_row in their connection list
        chk=get_indexes(p_row.Pore_index,pore1)
        chk2=get_indexes(p_row.Pore_index,pore2)
            #if (both list are empty go next pore) or (if less than 2 constrictions per pore), go next pore
        if (not chk and not chk2) or ((len(chk)+len(chk2))<2):
            i+=1
            continue
        chk=chk+chk2    #add both lists into one

            #for each of the constrictions connected to the pore p, get properties
        ctc_cnst=[]
        grID=[]
        cID=[]
        for j in chk:
            cdNb+=1                             #increase coordination number
            ctc_cnst=ctc_cnst+constr[j].contactPoint
            grID=grID+constr[j].grainID
            cID.append(constr[j].id)

        ### Check pore spheres : get grains in contact with pore sphere
            #for each interaction of the pore
        for ct in O.bodies[i+nb_filter+nb_cons].intrs():
            if ct.geom.PenetrationDepth>0 and (ct.id1<=nb_filter or ct.id2<=nb_filter):
                grID.append(min(ct.id1, ct.id2))
        
        ### Check grains : add the contact point between grains of grID (grains in contact with pore sphere or with constrictions )
            #Transform grID into array and get unique grains conected to constrictions
        grID=np.sort(np.unique(np.array(grID))) 
        
            #get grID centers and contact points between the them
        ctc_grains=[]
        gr_center=np.zeros([len(grID),3]) 
        for j in range(len(grID)):
            # grID is no longer int for some reason, force it into int
            gID=int(grID[j])
            #add grai center to array
            gr_center[j]=O.bodies[gID].state.pos
            #for each contact of grain j
            for k in O.bodies[gID].intrs():
                #if (contact has already been checked) or (there is no contact), jump it
                if gID>k.id1 or gID>k.id2 or k.geom.PenetrationDepth<=0:
                    continue
                #check if both ID's of the analysed contact belong to list of IDs of grains in contact with the constrictions
                bool_array = np.in1d(grID, [k.id1, k.id2])
                #if both grains belong to it, add the contact point into the matrix
                if np.size(grID[bool_array])==2:
                    ctc=k.geom.contactPoint #transform from Vector3 into normal list
                    ctc_grains.append([ctc[0],ctc[1],ctc[2]])
        
        ### Join above obtained vectors.
        #if ctc_grains is not empty, add it to constriction contact
        #if ctc_grains:
        #add gr_center and transform list in array
        ctc_cnst = np.array(ctc_cnst+ctc_grains)
        #add gr_center (already array)
        ctc_cnst=np.concatenate((ctc_cnst,gr_center)) 

        ### Calculate convex hull
        cHull= ConvexHull(ctc_cnst)

        ### Create pore object
        pr=Pore(p_row, cID, ctc_cnst, grID, cHull, cdNb)
        #create vtk file
        vtkPorePlot(cHull, pr, path)
        #append to list
        poreList.append(pr)
        i+=1

    print "Number of good pores :",len(poreList)
    print "Number of bad pores :",len(p_data)-len(poreList)

    #Crate a pore sphere vtk plot
    vtkSpherePlot(poreList, path, 'Pore_sph.vtk')
    with open(os.path.join(path,"pore_obj.pkl"), 'wb') as f:
        dill.dump(poreList, f)

    return poreList
#Results plot
def poreResultsPlot(poreL):
    #get the data from poreL object to be plot
    ratZ=np.zeros((len(poreL),1))
    ratV=np.zeros((len(poreL),1))
    r=range(len(poreL))
    for p in r:
        ratV[p]= poreL[p].volume_ratio
        ratZ[p]=poreL[p].coord_old/poreL[p].coord_new

    #Plot relative Volume
    plt.figure(0)
    plt.plot(r,ratV,'ro')
    plt.axhline(y = 1, color = 'k', linestyle = '-')
    plt.xlabel('ID')
    plt.ylabel('V ratio')
    plt.title('Pore V ratio')

    #Plot relative Z
    plt.figure(1)
    plt.plot(r,ratZ,'ro')
    plt.axhline(y = 1, color = 'k', linestyle = '-')
    plt.xlabel('ID')
    plt.ylabel('Z ratio')
    plt.title('Pore Z ratio')

    #Plot relative Volume histogram
    catV=[(ratV<0.2).sum(), np.logical_and(ratV>=0.2,ratV<1).sum(), np.logical_and(ratV>=1,ratV<10).sum(), (ratV>=10).sum()]
    catN= ['[0,0.2[', '[0.2,1[', '[1,10[', '[10,inf[']
    f=plt.figure(2)
    ax=f.add_axes([0,0,1,1])
    ax.bar(catN,catV)

    #Show plots
    plt.show()
#VTK Plot
def vtkSpherePlot(obj, path, fnm):
    #prepare data to vtkplot
    nbPt=len(obj)
    rad=np.zeros([nbPt,1])
    pos=np.zeros([nbPt,3])
    for c in range(nbPt):
        rad[c]=obj[c].radius
        pos[c]=obj[c].position

    #check if folder exists
    pth=os.path.join(path,'poreVtk')
    if not os.path.exists(pth):
        os.mkdir(pth)
    #Create or overwrite file ('w' option)
    fnm=os.path.join(pth,fnm)
    f = open(fnm, 'w'); 
    # VTK files contain five major parts
    # 1. VTK DataFile Version
    f.write('# vtk DataFile Version 2.0\n')
    # 2. Title
    f.write('Pore VTK draw from Python\n')
    # 3. The format of data
    f.write('ASCII\n');
    # 4. Type of Dataset
    f.write('DATASET UNSTRUCTURED_GRID\n')
    # 5. Describe the data set
    #   Save points
    f.write('POINTS {:d} float\n'.format(nbPt));
    for i in range(nbPt):
        f.write("{:.8f} {:.8f} {:.8f} \n".format(pos[i,0], pos[i,1], pos[i,2]));

    #   Save scalars
    f.write('\nPOINT_DATA {:d}'.format(nbPt));
    f.write('\nSCALARS Radius float 1\n');
    f.write('LOOKUP_TABLE default\n');
    for i in range(nbPt):
        f.write("{:.8f}\n".format(rad[i,0]))   
def vtkPorePlot(cHull, pore, path):
    #prepare data to vtkplot
    nbPt=cHull.points.shape[0] #nb of points
    nbTri=cHull.simplices.shape[0] #nb of triangles
    
    #check if folder exists
    pth=os.path.join(path,"poreVtk")
    if not os.path.exists(pth):
        os.mkdir(pth)
    #Create or overwrite file ('w' option)
    fnm=os.path.join(pth,"Pore_surf_{:d}.vtk".format(pore.id))
    f = open(fnm, 'w'); 
    # VTK files contain five major parts
    # 1. VTK DataFile Version
    f.write('# vtk DataFile Version 2.0\n')
    # 2. Title
    f.write('Pore VTK draw from Python\n')
    # 3. The format of data
    f.write('ASCII\n');
    # 4. Type of Dataset
    f.write('DATASET POLYDATA\n');
    # 5. Describe the data set
    #   Save points
    f.write('POINTS {:d} float\n'.format(nbPt))
    for i in range(nbPt):
        k=cHull.points[i]
        f.write("{:.8f} {:.8f} {:.8f} \n".format(k[0],k[1],k[2]))

    #   Save the triangles
    f.write("\nPOLYGONS {:d} {:d}\n".format(nbTri,4*nbTri))
    for i in range(nbTri):
        k=cHull.simplices[i]
        f.write("3 {:d} {:d} {:d}\n".format(k[0],k[1],k[2]))

    #   Face scalar
    f.write('\nCELL_DATA {:d}\n'.format(nbTri))
    f.write("SCALARS Volume_Ratio float 1\n")
    f.write('LOOKUP_TABLE default\n')
    for i in range(nbTri):
        f.write("{:.5f}\n".format(pore.volume_ratio))
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
    #elif not os.path.exists(os.path.join(path,voxel_fnm+'_SURF.dat')):
    #    print("File "+voxel_fnm+"_SURF.dat was not found.")
    #    val=True
    
    if val:
        print("Check if pnExtract was executed beforehand.")
        return val

    if not os.path.exists(os.path.join(path,'axis_list.npy')):
        print("File axis_list.npy was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,'angle_list.npy')):
        print("File angle_list.npy was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,'position_list.npy')):
        print("File position_list.npy was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,'rx_list.npy')):
        print("File rx_list.npy was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,'ry_list.npy')):
        print("File ry_list.npy was not found.")
        val=True
    elif not os.path.exists(os.path.join(path,'rz_list.npy')):
        print("File rz_list.npy was not found.")
        val=True

    if val:
        print("Missing DEM data.")
    
    return val  
def grCtcList(O,nb_filter,path):
    #Return the contact lists between "real grains"
    ctcL=[]
    for c in O.interactions:
        if c.id1<nb_filter and c.id1<nb_filter:
            ct=c.geom.contactPoint
            ctcL.append([c.id1,c.id2,ct[0],ct[1],ct[2]])

    c_val=["id1","id2","x","y","z"]
    df=pd.DataFrame(data=ctcL,columns=c_val)
    df.to_csv(os.path.join(path,"grain_contact_list.dat"))

    return ctcL