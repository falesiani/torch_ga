
"""
Utility functions to plot lines, planes and points in 3D. they assume G(0,1,1,1)

"""

import numpy as np
# import meshplot as mp
import matplotlib.pyplot as plt
# from icecream import ic

from .utils import pga2line, pga2plane, pga2point, line2pga, plane2pga, point2pga
from .utils import cga2point, point2cga, cga2line

from .mv import MultiVector
# from .torch_ga import GeometricAlgebra
import torch

def t2np(x):
    if isinstance(x,torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def plot_line_pga(ga,tensor,ax=None,alpha = 1.0, **kargs):
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    if len(tensor.shape)==1: tensor=tensor.unsqueeze(0)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        if ga.p==3: ax = fig.add_subplot(projection='3d')   
        else: ax = fig.add_subplot()   
            
    points, direction = pga2line(ga,tensor)    
    # ic(points,direction)
    # orthogonal_shift, direction = plucker_to_line(orthogonal_shift, direction)
    for s,d in zip(t2np(points), t2np(direction)):        
        plot_line_xyz(s,d,ax, ga.p, alpha, **kargs) 
        
    if True:
        coords = t2np(points)
        for ki,xy in enumerate(coords):
            ax.text(*xy,f"{ki}")        

def plot_plane_pga(ga,tensor,ax=None, **kargs):
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    if len(tensor.shape)==1: tensor=tensor.unsqueeze(0)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')    
        if ga.p==3: ax = fig.add_subplot(projection='3d')   
        else: ax = fig.add_subplot()   
        
    points, direction = pga2plane(ga,tensor)
    if ga.p==3:
        xx, yy = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
    else:
        xx = np.linspace(-1,1,10)
    for d,n in zip(t2np(points), t2np(direction)):
        if ga.p==3: plot_plane_xyz(d,n,ax,xx,yy,**kargs)
        if ga.p==2: plot_plane_xy(d,n,ax,xx,**kargs)
        
    # coords = t2np(direction)
    # for ki,xy in enumerate(coords):
    #     ax.text(*xy,f"{ki}")              

def plot_point_pga(ga,tensor, ax=None, **kargs):
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    if len(tensor.shape)==1: tensor=tensor.unsqueeze(0)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')    
        if ga.p==3: ax = fig.add_subplot(projection='3d')   
        else: ax = fig.add_subplot()   
        
    point = pga2point(ga,tensor)
    ax.scatter(*t2np(point).T,**kargs)
    
    coords = t2np(point)
    for ki,xy in enumerate(coords):
        ax.text(*xy,f"{ki}")

# CGA

def plot_point_cga(ga,tensor, ax=None, **kargs):
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    if len(tensor.shape)==1: tensor=tensor.unsqueeze(0)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')    
        if ga.p==3: ax = fig.add_subplot(projection='3d')   
        else: ax = fig.add_subplot()   
        
    point = cga2point(ga,tensor)
    ax.scatter(*t2np(point).T,**kargs)
    
    coords = t2np(point)
    for ki,xy in enumerate(coords):
        ax.text(*xy,f"{ki}")


def plot_line_cga(ga,tensor,ax=None,alpha = 1.0, **kargs):
    if isinstance(tensor, MultiVector): tensor = tensor.tensor
    if len(tensor.shape)==1: tensor=tensor.unsqueeze(0)
    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')    
        if ga.p==3: ax = fig.add_subplot(projection='3d')   
        else: ax = fig.add_subplot()   
        
    points, direction = cga2line(ga,tensor)    
    # orthogonal_shift, direction = plucker_to_line(orthogonal_shift, direction)
    for s,d in zip(t2np(points), t2np(direction)):        
        plot_line_xyz(s,d,ax, ga.p, alpha, **kargs)    
        
    if True:
        coords = t2np(points)
        for ki,xy in enumerate(coords):
            ax.text(*xy,f"{ki}") 

# eculidean
def plot_plane_xyz(d,n,ax,xx,yy, **kargs): 
    xn,yn,zn = n
    z = (d-xx*xn-yy*yn)/zn
    ax.plot_surface(xx, yy, z, alpha=0.2,**kargs)
def plot_plane_xy(d,n,ax,xx, **kargs): 
    xn,zn = n
    # z = (d-xx*xn)/zn
    z = (d-xx*xn)/zn
    ax.plot(z, xx, alpha=0.2,**kargs)

def plot_line_xyz(s,d,ax,dim, alpha = 1.0, **kargs):    
    # ic(dim,s,d) 
    _line = [(s[_]-alpha*d[_],s[_]+alpha*d[_]) for _ in range(dim)]
    ax.plot(*_line,**kargs)
    
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to convert Plücker coordinates to direction vector and point on line
def plucker_to_line(p,q):
  d = np.cross(p, q)  # Direction vector
#   ic(d.shape,p.shape)
  v = (p.T / d[...,-1]).T  # Point on the line (assuming d[3] != 0)
#   v = ic(p.T / q[...,0]).T  # Point on the line (assuming d[3] != 0)
#   v = ic(p / abs(d).max(-1))  # Point on the line (assuming d[3] != 0)
# v = ic(p / d[...,-1])  # Point on the line (assuming d[3] != 0)
#   v = ic(p)  # Point on the line (assuming d[3] != 0)
  return d, v

def build_lines(d,v,alpha):
    t = alpha*np.linspace(-1, 1, 100)  # Range for parameterizing the line
    line_x = v[...,0] + t * d[...,0]
    line_y = v[...,1] + t * d[...,1]
    line_z = v[...,2] + t * d[...,2]
    return line_x, line_y, line_z 

def plot_line_xyz_(s,d,ax,dim, alpha = 1.0, **kargs):   
    d, v = plucker_to_line(-s,-d)  
    line_x, line_y, line_z = build_lines(d,v,alpha)
    # _line = [(s[_]-alpha*d[_],s[_]+alpha*d[_]) for _ in range(3)]
    # ax.plot(*_line,**kargs)
    # ax.plot_trisurf(line_x.T, line_y.T, line_z.T, cmap='viridis')  # Plot line with colormap
    ax.plot(line_x, line_y, line_z)  # Plot line with colormap

# # Define Plücker coordinates
# p = np.array([p1, p2, p3, p4, p5, p6])  # Replace with your desired values

# # Function to convert Plücker coordinates to direction vector and point on line
# def plucker_to_line(p):
#   d = np.cross(p[:3], p[3:])  # Direction vector
#   v = p[:3] / d[3]  # Point on the line (assuming d[3] != 0)
#   return d, v

# # Convert Plücker coordinates to direction vector and point
# d, v = plucker_to_line(p)

# # Create 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the line
# t = np.linspace(-1, 1, 100)  # Range for parameterizing the line
# line_x = v[0] + t * d[0]
# line_y = v[1] + t * d[1]
# line_z = v[2] + t * d[2]

# ax.plot_trisurf(line_x, line_y, line_z, cmap='viridis')  # Plot line with colormap

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Line in Plücker Coordinates')

# # plt.show()
    