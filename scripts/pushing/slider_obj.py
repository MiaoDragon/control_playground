#!/usr/bin/env python
# coding: utf-8

# # Define the object class for the slider
# The object should have three components: geometry, limit surface model, and motion cone angles.
# ## geometry
# many ways can be used to define the geometry. Geometry should include functions that can compute the normals, detect whether a point is on the surface or not, and sample points on the surface.
# Ways to implement geometry:
# - shape primitives
# - point cloud
# - grid
# - convex hull
# 
# ## limit surface model
# grid-based method: implemented
# polynomial methods: TODO
# ## motion cone angle
# determine the motion cone angles for each point on the surface.

# In[ ]:


import numpy as np
import utilities as utils


# In[ ]:


class Slider:
    def __init__(self, geometry, ls_model, motion_cone, pose):
        """
        root definition of Slider
        NOTE: the COM and coordinate system of the three components should be the same
        """
        self.geometry = geometry
        self.ls_model = ls_model
        self.motion_cone = motion_cone
        self.pose = pose  # init pose. All the three components work in the local frame
    def point_pusher_model(self, p, v_pusher):
        """
        given the pushing point p, the velocity of the pusher v_pusher, and the limit surface model H.
        determine the velocity of the slider. (other intermediate values such as frictions too)
        
        here we use a single pt p.
        TODO: future work can apply a batched version. Possibly add a new function for BATCH
        """
        H = self.ls_model
        # * obtain the friction cone at p
        normal = self.geometry.get_normal(p).reshape(-1)
        angle = self.motion_cone.get_angle(p)
        fl = utils.rot_mat_from_ang(-angle).dot(normal)
        fu = utils.rot_mat_from_ang(angle).dot(normal)
        
        # * check if it's sticking mode: find F such that the equation holds
        # check the normals to the convex hull faces, and find the one closest to v_pusher
        F = H.get_load_point_vel(p, v_pusher, v_err=1e-3, f_err=1e-1)
        if F is None:
            pass
        else:
            # check the force to remain in the cone
            f = np.array(F[:2])
            theta_fl = np.arctan2(fl[1], fl[0])
            theta_fu = np.arctan2(fu[1], fu[0])
            if theta_fu < theta_fl:
                theta_fu += np.pi*2
            theta_f = np.arctan2(f[1], f[0])
            if theta_f < theta_fl:
                theta_f += np.pi*2
            if theta_fl < theta_f and theta_f < theta_fu:
                return F
            
        # * otherwise it's in the sliding mode
        # f = fl or fu
        F = np.array([fl[0], fl[1], p[0]*fl[1]-p[1]*fl[0]])
        Fl = np.array(F)
        V = H.get_twist(F.reshape((1,-1)))[0]
        # verify the difference
        v = np.zeros((2))
        v[0] = V[0] - p[1]*V[2]
        v[1] = V[1] + p[0]*V[2]
        # check if it's fit
        normal = (fl+fu)/2
        flt = fl - fl.dot(normal)*normal
        vt = v - v.dot(normal)*normal  # get the tangent velocity
        if vt.dot(flt) >= 0:
            return Fl
        else:
            F = np.array([fu[0], fu[1], p[0]*fu[1]-p[1]*fu[0]])
            Fu = np.array(F)
            return Fu

