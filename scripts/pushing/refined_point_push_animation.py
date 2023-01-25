#!/usr/bin/env python
# coding: utf-8

# # Point-contact simulation for pushing
# ### Free moving point contact (following a pre-defined traj), simulate the response

# In[1]:


from refined_push_sim import *
from slider_geometry import *
from motion_cone import *
from slider_obj import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from IPython import display

# In[2]:


class FreePointSimulator:
    def __init__(self, slider, transform, point_transform, point_vel_list, dt):
        """
        H: the limit surface model of the object
        transform: the initial transform of the slider COM (x,y,theta)
        point_transform: initial transform of the point (x,y)
        point_vel_list: the list of velocities of the point
        """
        self.slider = slider
        self.transform = transform
        self.point_transform = point_transform
        self.point_vel_list = point_vel_list
        self.dt = dt
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        xmin = -slider.geometry.w/2*1.5
        xmax = slider.geometry.w/2*1.5
        ymin = -slider.geometry.h/2*1.5
        ymax = slider.geometry.h/2*1.5
        self.ax.axis([-slider.geometry.w/2*1.5, slider.geometry.w/2*1.5, 
                      -slider.geometry.h/2*1.5, slider.geometry.h/2*1.5])
        self.ax.set_xticks(np.arange(xmin, xmax, slider.geometry.resols[0]))
        self.ax.set_yticks(np.arange(ymin, ymax, slider.geometry.resols[1]))
        self.ax.set_aspect('equal', adjustable='box')

    def init_anim(self):
        slider = self.slider
        alpha = slider.ls_model.friction_coeff*slider.ls_model.pressure_dist
        alpha = alpha / alpha.max()    
        
        dx = slider.geometry.w / slider.geometry.grid.shape[0]
        dy = slider.geometry.h / slider.geometry.grid.shape[1]
        patches = []
        for i in range(slider.geometry.grid.shape[0]):
            for j in range(slider.geometry.grid.shape[1]):
                x = -slider.geometry.w/2 + i*dx
                y = -slider.geometry.h/2 + j*dy
                slider_ij = plt.Rectangle((self.transform[0]+x,self.transform[1]+y), 
                                          dx, dy, color='gray', alpha=alpha[i,j])
                # we approxiate each rectangular grid by a circle to be symmetric
                t = mpl.transforms.Affine2D().rotate_deg_around(self.transform[0], self.transform[1],                                         transform[2]/np.pi * 180) + self.ax.transData
                slider_ij.set_transform(t)
                # TODO: adjust color based on value
                self.ax.add_patch(slider_ij)
                patches.append(slider_ij)
        pusher = plt.Circle((self.point_transform[0], self.point_transform[1]), 0.01, color='green')
        patches.append(pusher)
        self.ax.add_patch(pusher)
        self.patches = patches
        return patches
    def transform_to_mat(self, transform):
        mat = np.eye(3)
        mat[0,0] = np.cos(transform[2])
        mat[0,1] = -np.sin(transform[2])
        mat[1,0] = np.sin(transform[2])
        mat[1,1] = np.cos(transform[2])
        mat[0,2] = transform[0]
        mat[1,2] = transform[1]
        return mat
    def step(self, i):
        slider = self.slider
        # * assuming rectangle
        vel = self.point_vel_list[i]
        # project the contact point in the frame of slider
        mat = self.transform_to_mat(self.transform)
        pt_transform = np.zeros((3))
        pt_transform[2] = 1
        pt_transform[:2] = self.point_transform
        pt_in_slider = np.linalg.inv(mat).dot(pt_transform)[:2]
        
        rot_in_slider = mat[:2,:2]
        rot_in_slider = np.linalg.inv(rot_in_slider)
        def get_rotation_mat(ang):
            # given ang in radians, get the rotation matrix
            return np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])        

        if slider.geometry.check_pt_surface(pt_in_slider):
            norm = slider.geometry.get_normal(pt_in_slider)[0]
            # contact
            vel_in_slider = rot_in_slider.dot(vel)
            # TODO: in point_pusher_model, finish up to get the V and v            
            F = slider.point_pusher_model(pt_in_slider, vel_in_slider)
#             F = point_pusher_model(pt_in_slider, vel_in_slider,
#                                fl, fu, self.H, vis=True)
            V = slider.ls_model.get_twist(F.reshape((1,-1)))[0]
            # make sure the size is right
            v_p = np.array([V[0]-pt_in_slider[1]*V[2], V[1]+pt_in_slider[0]*V[2]])
            norm_vel = vel_in_slider.dot(norm)
            scale = norm_vel / v_p.dot(norm)
            V = V * scale
            # get the linear velocity
            V[:2] = mat[:2,:2].dot(V[:2])
            self.transform += V * self.dt
            self.point_transform += vel*self.dt
            print('V: ', V)
        else:
            # not in contact
            self.point_transform += vel*self.dt

        # update the transform
        dx = slider.geometry.w / slider.geometry.grid.shape[0]
        dy = slider.geometry.h / slider.geometry.grid.shape[1]
        patches = []
        for i in range(slider.geometry.grid.shape[0]):
            for j in range(slider.geometry.grid.shape[1]):
                x = -slider.geometry.w/2 + i*dx
                y = -slider.geometry.h/2 + j*dy
                idx = i*slider.geometry.grid.shape[1]+j
                self.patches[idx].set_xy([self.transform[0]+x,self.transform[1]+y])
                t = mpl.transforms.Affine2D().rotate_deg_around(self.transform[0], self.transform[1],                                         self.transform[2]/np.pi * 180) + self.ax.transData
                self.patches[idx].set_transform(t)
        self.patches[-1].center = self.point_transform[0],self.point_transform[1]
        return self.patches
    def anim(self):
        ani = FuncAnimation(self.fig, self.step, frames=len(self.point_vel_list),
                            init_func=self.init_anim, interval=int(1000*dt), blit=False)
        video = ani.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()           



class FreePointMoveSimulator:
    def __init__(self, slider, transform, point_tfs_in_slider, point_vels_in_slider, dt):
        """
        H: the limit surface model of the object
        transform: the initial transform of the slider COM (x,y,theta)
        point_transform: initial transform of the point (x,y)
        point_vel_list: the list of velocities of the point
        """
        self.slider = slider
        self.transform = transform
        self.point_tfs_in_slider = point_tfs_in_slider
        self.point_vels_in_slider = point_vels_in_slider
        self.dt = dt
        self.fig, self.ax = plt.subplots(figsize=(8, 8*slider.geometry.h/slider.geometry.w))
        xmin = -slider.geometry.w/2*1.5
        xmax = slider.geometry.w/2*1.5
        ymin = -slider.geometry.h/2*1.5
        ymax = slider.geometry.h/2*1.5
        self.ax.axis([-slider.geometry.w/2*1.5, slider.geometry.w/2*1.5, 
                      -slider.geometry.h/2*1.5, slider.geometry.h/2*1.5])
        self.ax.set_xticks(np.arange(xmin, xmax, slider.geometry.resols[0]))
        self.ax.set_yticks(np.arange(ymin, ymax, slider.geometry.resols[1]))
        self.ax.set_aspect('equal', adjustable='box')

    def init_anim(self):
        slider = self.slider
        alpha = slider.ls_model.friction_coeff*slider.ls_model.pressure_dist
        alpha = alpha / alpha.max()    
        
        dx = slider.geometry.w / slider.geometry.grid.shape[0]
        dy = slider.geometry.h / slider.geometry.grid.shape[1]
        patches = []

        transform = self.transform
        mat = utils.transform_vec_to_mat(transform)
        point_transform = mat[:2,:2].dot(self.point_tfs_in_slider[0]) + mat[:2,2]
        for i in range(slider.geometry.grid.shape[0]):
            for j in range(slider.geometry.grid.shape[1]):
                x = -slider.geometry.w/2 + i*dx
                y = -slider.geometry.h/2 + j*dy
                slider_ij = plt.Rectangle((self.transform[0]+x,self.transform[1]+y), 
                                          dx, dy, color='gray', alpha=alpha[i,j])
                # we approxiate each rectangular grid by a circle to be symmetric
                t = mpl.transforms.Affine2D().rotate_deg_around(transform[0], transform[1],transform[2]/np.pi * 180) + self.ax.transData
                slider_ij.set_transform(t)
                # TODO: adjust color based on value
                self.ax.add_patch(slider_ij)
                patches.append(slider_ij)
        pusher = plt.Circle((point_transform[0], point_transform[1]), 0.01, color='green')
        patches.append(pusher)
        self.ax.add_patch(pusher)
        self.patches = patches
        return patches
    def transform_to_mat(self, transform):
        mat = np.eye(3)
        mat[0,0] = np.cos(transform[2])
        mat[0,1] = -np.sin(transform[2])
        mat[1,0] = np.sin(transform[2])
        mat[1,1] = np.cos(transform[2])
        mat[0,2] = transform[0]
        mat[1,2] = transform[1]
        return mat
    def step(self, i):
        slider = self.slider
        # * assuming rectangle
        vel_in_slider = self.point_vels_in_slider[i]
        # project the contact point in the frame of slider
        mat = self.transform_to_mat(self.transform)
        transform_i = mat[:2,:2].dot(self.point_tfs_in_slider[i]) + mat[:2,2]
        pt_in_slider = self.point_tfs_in_slider[i]

        vel = mat[:2,:2].dot(vel_in_slider)

        rot_in_slider = mat[:2,:2]
        rot_in_slider = np.linalg.inv(rot_in_slider)

        if slider.geometry.check_pt_surface(pt_in_slider):
            norm = slider.geometry.get_normal(pt_in_slider)[0]
            # contact
            # TODO: in point_pusher_model, finish up to get the V and v            
            F = slider.point_pusher_model(pt_in_slider, vel_in_slider)
#             F = point_pusher_model(pt_in_slider, vel_in_slider,
#                                fl, fu, self.H, vis=True)
            V = slider.ls_model.get_twist(F.reshape((1,-1)))[0]
            # make sure the size is right
            v_p = np.array([V[0]-pt_in_slider[1]*V[2], V[1]+pt_in_slider[0]*V[2]])
            norm_vel = vel_in_slider.dot(norm)
            scale = norm_vel / v_p.dot(norm)
            V = V * scale
            # get the linear velocity
            V[:2] = mat[:2,:2].dot(V[:2])
            self.transform += V * self.dt
            transform_i += vel*self.dt
            print('V: ', V)
        else:
            # not in contact
            transform_i += vel*self.dt

        # update the transform
        dx = slider.geometry.w / slider.geometry.grid.shape[0]
        dy = slider.geometry.h / slider.geometry.grid.shape[1]
        patches = []
        for i in range(slider.geometry.grid.shape[0]):
            for j in range(slider.geometry.grid.shape[1]):
                x = -slider.geometry.w/2 + i*dx
                y = -slider.geometry.h/2 + j*dy
                idx = i*slider.geometry.grid.shape[1]+j
                self.patches[idx].set_xy([self.transform[0]+x,self.transform[1]+y])
                t = mpl.transforms.Affine2D().rotate_deg_around(self.transform[0], self.transform[1],                                         self.transform[2]/np.pi * 180) + self.ax.transData
                self.patches[idx].set_transform(t)
        self.patches[-1].center = transform_i[0],transform_i[1]
        return self.patches
    def anim(self):
        ani = FuncAnimation(self.fig, self.step, frames=len(self.point_tfs_in_slider),
                            init_func=self.init_anim, interval=int(1000*self.dt), blit=False)
        video = ani.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()           

# %%
