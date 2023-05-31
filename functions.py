# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:36:31 2021

@author: edgar
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.ckdtree import cKDTree
#from mpl_toolkits.mplot3d import Axes3D
#import random as ran

def GetPoreData(path):
    data=pd.read_csv(path+'_node2.dat',delimiter='\s+',header=None,names=["Pore_index","Pore_volume","Pore_radius","Pore_Shape_factor","Pore_Equivalent_Diameter"])
    data2=pd.read_csv(path+'_node1.dat',delimiter='\s+',header=0,usecols=range(5),names=["Pore_index","x","y","z","Pore_Coordination_number"])
    test=data2.merge(data)
    test.Pore_Equivalent_Diameter=pow((3/(4*np.pi)*test.Pore_volume),1/3)*2

    return test
    
def GetThroatData(path):
    data=pd.read_csv(path+'_link2.dat',delimiter='\s+',header=None,names=["Throat_index","Pore_1_index","Pore_2_index","Length_of_Pore_1","Length_of_Pore_2","Length_of_Throat","Throat_Volume","Throat_Clay_Volume"])
    data2=pd.read_csv(path+'_link1.dat',delimiter='\s+',header=0,names=["Throat_index","Pore_1_index","Pore_2_index","t_x",'t_y',"t_z","Throat_Radius","Throat_Shape_Factor","Throat_Total_Length"])
    test=data2.merge(data)
    
    return test

def UpdateCoordNumber(data_pore,data_throat):
    data_pore2=data_pore.copy()
    index=data_pore2.Pore_index.values
    coord=np.zeros(len(index))
    for i in range(len(index)):
        coord[i]=((data_throat.Pore_1_index==index[i])|(data_throat.Pore_2_index==index[i])).sum()
    data_pore2.loc[:,"Pore_Coordination_number"]=coord
    return(data_pore2)

def DeleteBorderElements(data_pore,data_throat,alpha):
     
    #Load data and get size of the box
    
    test_Pore=data_pore
    test_Throat=data_throat
    mx=min(test_Pore.x)
    Mx=max(test_Pore.x)
    my=min(test_Pore.y)
    My=max(test_Pore.y)
    mz=min(test_Pore.z)
    Mz=max(test_Pore.z)
    
    #Get index of pores located between 0 and N*alpha/2 and (1-alpha/2)*N  N for every axis
    
    ind=((test_Pore.x<((alpha/2)*(Mx-mx)+mx)) | (test_Pore.x>((1-alpha/2)*(Mx-mx)+mx)) 
     | (test_Pore.y<((alpha/2)*(My-my)+my)) | (test_Pore.y>((1-alpha/2)*(My-my)+my))
     | (test_Pore.z<((alpha/2)*(Mz-mz)+mz)) | (test_Pore.z>((1-alpha/2)*(Mz-mz)+mz))
    )
    ind_Pore=np.append(test_Pore.loc[ind].Pore_index.values,[-1,0]) #Throat from border can be linked to only 1 Pore, 2nd Pore is identified by id=-1 or 0
    
    #Get index of Throats that were linked to any pore identified previously
    
    ind_throat=test_Throat.Pore_1_index.isin(ind_Pore)|test_Throat.Pore_2_index.isin(ind_Pore)
    
    data_Throats=test_Throat.loc[~ind_throat]
    
    data_Pore=test_Pore.loc[~ind]

    data_Pore=UpdateCoordNumber(data_Pore,data_Throats)
    
    return [data_Pore,data_Throats]       

def N_components(points,alpha):
    mx=min(points[:,0])
    my=min(points[:,1])
    mz=min(points[:,2])
    Mx=max(points[:,0])
    My=max(points[:,1])
    Mz=max(points[:,2])
    
    if (len(points)<5):
        N_comp=0
    
    else:    
    
        grid=np.zeros([Mx-mx+1,My-my+1,Mz-mz+1])

        points2=[points[:,0]-mx,points[:,1]-my,points[:,2]-mz]

        for i in range(len(points)):
            grid[points2[0][i],points2[1][i],points2[2][i]]=1
        cs = np.argwhere(grid > 0)

        # build k-d tree
        kdt = cKDTree(cs)
        edges = kdt.query_pairs(np.sqrt(1))

        # create graph
        G = nx.from_edgelist(edges)

        # find connected components
        ccs = nx.connected_components(G)
        node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
        N_comp=nx.number_connected_components(G)

        # Discard small components relative to the main component

        set1=set(node_component.values())
        tab=np.zeros(len(set1))
        k=0
        for value in set1:
            tab[k]=sum(map((k).__eq__, node_component.values()))
            k+=1
        N_comp-=(tab<tab[0]*alpha).sum()
    
    return N_comp

def GetThroatInfo(path,alpha):
    
    surf=pd.read_csv(path+'_SURF.dat',sep='\s+',names=["t_id","x","y","z"])
    t_min=min(surf.loc[:,"t_id"])
    t_max=max(surf.loc[:,"t_id"])
    N=len(surf)
    final=pd.DataFrame(index=np.arange(0,t_max),columns=["t_id","norm_vector_x","norm_vector_y","norm_vector_z"])
    #removed ,"N_components" atribute cause N_components functon was failing
    index=surf.t_id.values
    uniq=np.unique(index)
    mx=len(uniq)
    pct=0
    print("Starting throat normal calculation")  
    for i in range(mx):
        temp=surf[index==np.unique(index)[i]].values[:,1:4]#Get coordinates
        
        #This give the normal vector by computing SVD to get mean plane through our points
        U, D, V = np.linalg.svd(np.transpose(temp)-np.mean(np.transpose(temp),axis=1,keepdims=True),full_matrices=False)
        final.loc[i,["norm_vector_x","norm_vector_y","norm_vector_z"]]=U[:,-1]
        
        final.loc[i,"t_id"]=i+1
        if i*100/mx>pct+5:
            pct=pct+5
            print("Execution at "+ str(pct) +"%")  
        #final.loc[i,"N_components"]=N_components(temp,alpha)
    
    return final

def GetIndexAndDataFromPoresToReassign(data_pore,data_throat,Min,Max,criterion): #criterion="closest" or "biggest"

    ind=((data_pore.Pore_Equivalent_Diameter>Min) & (data_pore.Pore_Equivalent_Diameter<Max)) #Get pore index from pore in the wanted range
    index=data_pore[ind].Pore_index.values

    Pore_connection=pd.DataFrame(index=np.arange(0,len(index)),columns=["Pore_index","Reassigned_to_Pore_id","Smallest_euclidean_distance","Mean_Pore_E_Diameter","Neighboor_chain_length"])
    
    for i in range (len(index)):
       
        #Get index of all Pores that have a shared throats with the considered Pore
        
        Pore_connection.loc[i,"Pore_index"]=index[i]
        
        P1=data_throat[data_throat.Pore_2_index==index[i]].Pore_1_index.values
        P2=data_throat[data_throat.Pore_1_index==index[i]].Pore_2_index.values
        P=np.concatenate([P1,P2])
        
        #We don't care about Pore index 0 and -1 that are borders element (not real pores)
        
        #It is possible that we couldn't that we couldn't find any neighoor, we then assign -1
        if (len(P)==0):
            Pore_connection.loc[i,"Reassigned_to_Pore_id"]=-1
            Pore_connection.loc[i,"Neighboor_chain_length"]=-1
            
        else:
            
        
            rm=np.argwhere((P==0)|(P==-1)).flatten()
            P=np.delete(P,rm)

            temp=data_pore.Pore_index==index[i]

            #Get info and the considered pore

            point1=data_pore[temp].loc[:,["x","y","z"]].values
            #diam1=data_pore[temp].loc[:,"Pore_Equivalent_Diameter"].values

            #Get info of the Pores sharing a throat

            temp1=data_pore.Pore_index.isin(P)                 
            points=data_pore[temp1].loc[:,["x","y","z"]].values
            #diams=data_pore[temp1].loc[:,"Pore_Equivalent_Diameter"].values

            N=len(P)
            dist=np.zeros(N)

            #Get euclidean distance

            for k in range(N):
                dist[k]=np.linalg.norm(point1-points[k])
            mindist=min(dist)
            indmin=P[np.argmin(dist)]

            #Write results of Pore and it's neighboors
            
            Pore_connection.loc[i,"Smallest_euclidean_distance"]=mindist
            mean=data_pore[temp1].loc[:,"Pore_Equivalent_Diameter"].values.mean()
            Pore_connection.loc[i,"Mean_Pore_E_Diameter"]=mean

            #It's possible that all direct Pores sharing a throat are in consideration for deletion, so we extend to undirect neighboors
            #Chain will keep trace of the degree or neighboor (chain=0 is direct neighboor)

            chain=0
            while((~np.isin(P,index)).sum()==0 and len(P)!=0):


                P1=data_throat[data_throat.Pore_2_index.isin(P1)].Pore_1_index.values
                P2=data_throat[data_throat.Pore_1_index.isin(P2)].Pore_2_index.values
                P=np.concatenate([P1,P2])
                P=np.unique(P)

                rm=np.argwhere((P==0)|(P==-1)).flatten()
                P=np.delete(P,rm)
                chain+=1

            #It is possible that we couldn't that we couldn't find any neighoor, we then assign -1
            if (len(P)==0):
                Pore_connection.loc[i,"Reassigned_to_Pore_id"]=-1
                Pore_connection.loc[i,"Neighboor_chain_length"]=-1 
                
            else:

                P=P[~np.isin(P,index)] #To reassign, we only consider Pores that will not be reassigned
                temp1=data_pore.Pore_index.isin(P)                 
                points=data_pore[temp1].loc[:,["x","y","z"]].values
                #diams=data_pore[temp1].loc[:,"Pore_Equivalent_Diameter"].values

                N=len(P)
                dist=np.zeros(N)

                for k in range(N):
                    dist[k]=np.linalg.norm(point1-points[k])
                mindist=min(dist)
                indmin=P[np.argmin(dist)]  

                Pore_connection.loc[i,"Neighboor_chain_length"]=chain

                if (criterion=="closest"):
                    Pore_connection.loc[i,"Reassigned_to_Pore_id"]=indmin


                elif (criterion=="biggest"):
                    #big=max(data_pore[temp1].loc[:,"Pore_Equivalent_Diameter"].values)
                    indmax=P[np.argmax(data_pore[temp1].loc[:,"Pore_Equivalent_Diameter"].values)]
                    Pore_connection.loc[i,"Reassigned_to_Pore_id"]=indmax

                else:
                    print("Error, only criterion available are closest and biggest")
                    break
                

    return Pore_connection

def ReassignPoreAndThroats(Pore_connection,data_pore,data_throat):
    
    #Start by deleting all pores that we couldn't reassign properly (id=-1)
    
    ind_rm=Pore_connection[Pore_connection.Reassigned_to_Pore_id<=0].Pore_index
    Pore_connection=Pore_connection[Pore_connection.Reassigned_to_Pore_id>0]

    index=np.unique(Pore_connection.Reassigned_to_Pore_id)

    #copy old data and pre remove pores and throats that we can't reassign
    data_pore_reassigned=data_pore.copy(deep=True)
    data_pore_reassigned=data_pore_reassigned[~data_pore_reassigned.Pore_index.isin(ind_rm)]
    data_throat_reassigned=data_throat.copy(deep=True)
    data_throat_reassigned=data_throat_reassigned[~data_throat_reassigned.Pore_1_index.isin(ind_rm)]
    data_throat_reassigned=data_throat_reassigned[~data_throat_reassigned.Pore_2_index.isin(ind_rm)]
    
    
    #We reassign pores according to the index from GetIndexAndDataFromPoresToReassign function (Pore_connection)
    data_pore_reassigned.loc[Pore_connection.Pore_index.values-1]=data_pore_reassigned.loc[Pore_connection.Reassigned_to_Pore_id.values-1].values

    #We still have some duplicates from different pores being reassigned to the same pores, so we delete those duplicates
    data_pore_reassigned.drop_duplicates(subset=['Pore_index'],inplace=True)

    #Now, we loop over old Pores data to get the volume and coordination number so we can add it to the new pore
    for i in range(len(index)):
        index_pore=Pore_connection[Pore_connection.Reassigned_to_Pore_id==index[i]].Pore_index
        volume=data_pore[data_pore.Pore_index.isin(index_pore)].Pore_volume.sum()

        data_pore_reassigned.loc[index[0]-1,"Pore_volume"]+=volume

    data_pore_reassigned.Pore_Equivalent_Diameter=pow((3/(4*np.pi)*data_pore_reassigned.Pore_volume),1/3)*2

    #We also update the throat connections
    for i in range(len(Pore_connection)):

        P1=data_throat_reassigned.Pore_1_index==Pore_connection.Pore_index.values[i]
        P2=data_throat_reassigned.Pore_2_index==Pore_connection.Pore_index.values[i]

        data_throat_reassigned.loc[data_throat_reassigned[P1].index,"Pore_1_index"]=Pore_connection.Reassigned_to_Pore_id.values[i]
        data_throat_reassigned.loc[data_throat_reassigned[P2].index,"Pore_2_index"]=Pore_connection.Reassigned_to_Pore_id.values[i]

    #We still didn't removed the throats that linked a pore to the reassigned pore.
    #But now they can easily be identified, they have pore_1_id == pore_2_id since we updated the throats previously
    
    data_throat_reassigned=data_throat_reassigned[~(data_throat_reassigned.Pore_1_index==data_throat_reassigned.Pore_2_index)]
    
        
    #We still need to update the coordination number of each Pore
    data_pore_reassigned=UpdateCoordNumber(data_pore_reassigned,data_throat_reassigned)
    
    return[data_pore_reassigned,data_throat_reassigned]
   
def plotthroat(vox_size,surf,data_throat,tab_id,alpha):
    for j in tab_id:

        #surf=pd.read_csv(path+'_SURF.dat',sep='\s+',names=["t_id","x","y","z"])
        index=surf.t_id.values
        uniq=np.unique(index)
        temp=surf[index==np.unique(index)[j-1]].values[:,1:4]
        points=temp

        mx=min(points[:,0])
        my=min(points[:,1])
        mz=min(points[:,2])
        Mx=max(points[:,0])
        My=max(points[:,1])
        Mz=max(points[:,2])

        grid=np.zeros([Mx-mx+1,My-my+1,Mz-mz+1])

        points2=[points[:,0]-mx,points[:,1]-my,points[:,2]-mz]

        for i in range(len(points)):
            grid[points2[0][i],points2[1][i],points2[2][i]]=1

        # create data
        data2=grid;

        # find coordinates
        cs = np.argwhere(data2 > 0)

        # build k-d tree
        kdt = cKDTree(cs)
        edges = kdt.query_pairs(np.sqrt(1))

        # create graph
        G = nx.from_edgelist(edges)

        # find connected components
        ccs = nx.connected_components(G)
        node_component = {v:k for k,vs in enumerate(ccs) for v in vs}

        # visualize
        df = pd.DataFrame(cs, columns=['x','y','z'])
        df['c'] = pd.Series(node_component)

        # to include single-node connected components
        df.loc[df['c'].isna(), 'c'] = df.loc[df['c'].isna(), 'c'].isna().cumsum() + df['c'].max()

        # to delete components with not enough value (noises)
        n_comp=np.unique(df['c'].values)
        tab=np.zeros(len(n_comp),'int32')
        for k in range(len(n_comp)):
            tab[k]=(df['c'].values==n_comp[k]).sum()
       
        princ_comp=np.argmax(tab)
        delete=n_comp[tab<tab[princ_comp]*alpha]  #If components have less than 10% of the size of the princ_component, we delete it
        df=df[~df['c'].isin(delete)]

        fig = plt.figure(figsize=(max(Mx-mx,My-my,Mz-mz)+1,max(Mx-mx,My-my,Mz-mz)+1))
        ax = fig.add_subplot(111, projection='3d')

       # c=(data_throat.loc[j-1,"t_x"],data_throat.loc[j-1,"t_y"],data_throat.loc[j-1,"t_z"])
       # r=data_throat.loc[j-1,"Throat_Radius"]
       
        c=(data_throat[data_throat.Throat_index==(j)].t_x.values,data_throat[data_throat.Throat_index==(j)].t_y.values,data_throat[data_throat.Throat_index==(j)].t_z.values)
        r=data_throat[data_throat.Throat_index==(j)].Throat_Radius.values

        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)

        ax.plot_wireframe(x+c[0], y+c[1], z+c[2], color='b', alpha=0.8)

        cmhot = plt.get_cmap("Dark2")
        ax.scatter((df['x']+mx)*vox_size,(df['y']+my)*vox_size, (df['z']+mz)*vox_size, c=df['c'], s=20, cmap=cmhot)

        def set_axes_equal(ax):
            '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
            cubes as cubes, etc..  This is one possible solution to Matplotlib's
            ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

            Input
              ax: a matplotlib axis, e.g., as output from plt.gca().
            '''

            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5*max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        set_axes_equal(ax)

        plt.show()
    
