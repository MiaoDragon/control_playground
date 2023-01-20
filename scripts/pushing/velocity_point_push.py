"""
We use the limit cycle to predict the motion of the slider given velocity of the pusher.
This is implementing a recent paper by Alberto

Object models to be considered:
1) cylinders
2) blocks
"""
import numpy as np
class Geometry:
    def __init__(self, shape, mins, maxs, resols):
        self.shape = shape
        self.grid_size = (maxs-mins)/resols
        self.grid_size = np.ceil(self.grid_size).astype(int)
        self.mins = mins
        self.maxs = maxs
        self.resols = resols
        self.grid = np.zeros(self.grid_size).astype(int)  # an occupancy grid
        self.grid_x, self.grid_y = np.indices(self.grid_size)
        self.grid_x = self.resols[0] * (self.grid_x+0.5) + self.mins[0]
        self.grid_y = self.resols[1] * (self.grid_y+0.5) + self.mins[1]

    def update_grid(self):
        # obtain the occupancy grid based on the shape
        pass
    def local_inside(self, pt):
        pass

class CylinderShape(Geometry):
    def __init__(self, shape, mins, maxs, resols):
        super().__init__(shape, mins, maxs, resols)
        self.update_grid()
    def update_grid(self):
        # check the points if they're inside the grid
        inside = np.sqrt(self.grid_x ** 2 + self.grid_y ** 2)
        inside = inside <= self.shape
        self.grid = inside
        print('after update_grid')
    def local_inside(self, pt):
        return np.sqrt(pt[0]**2 + pt[1]**2) <= self.shape
class BlockShape(Geometry):
    def __init__(self, shape, mins, maxs, resols):
        super().__init__(shape, mins, maxs, resols)
        self.update_grid()
    def update_grid(self):
        # check the points if they're inside the grid
        inside = (self.grid_x <= self.shape[0]/2) & (self.grid_y <= self.shape[1]/2) & \
                (self.grid_x >= -self.shape[0]/2) & (self.grid_y >= -self.shape[1]/2)
        self.grid = inside
    def local_inside(self, pt):
        inside = (pt[0] <= self.shape[0]/2) & (pt[1] <= self.shape[1]/2) & \
                (pt[0] >= -self.shape[0]/2) & (pt[1] >= -self.shape[1]/2)
        return inside
        

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class Slider:
    def __init__(self, transform, shape):
        # transform: x,y,theta
        # shape: cylinder is radius, block is (height, width)
        self.transform = transform
        x,y,theta = transform
        tf_mat = [[np.cos(theta), -np.sin(theta),x], [np.sin(theta), np.cos(theta),y],[0,0,1]]
        tf_mat = np.array(tf_mat)
        self.tf_mat = tf_mat
        self.shape = shape
        self.tran_vel = np.zeros((2))
        self.rot_vel = 0
        self.type = 'none'
        self.hull = None
    def global_inside(self, pt):
        tf_mat = np.linalg.inv(self.tf_mat)
        pt = tf_mat[:2,:2].dot(pt) + tf_mat[:2,2]
        return self.geo.local_inside(pt)
    def get_tf_mat(self, transform):
        x,y,theta = transform
        tf_mat = [[np.cos(theta), -np.sin(theta),x], [np.sin(theta), np.cos(theta),y],[0,0,1]]
        tf_mat = np.array(tf_mat)
        return tf_mat
    def update_transform(self, new_transform):
        self.transform = new_transform
        self.tf_mat = self.get_tf_mat(new_transform)
    def update_vel(self, tran_vel, rot_vel):
        self.tran_vel = tran_vel
        self.rot_vel = rot_vel
    def get_contact_pt(self, contact_angle):
        pass
    def get_contact_angle_from_global(self, contact_pt):
        # first project contact pt to the local frame
        tf_mat_local = np.linalg.inv(self.tf_mat)
        contact_pt = tf_mat_local[:2,:2].dot(contact_pt) + tf_mat_local[:2,2]
        return self.get_contact_angle_from_local(contact_pt)
    def get_contact_angle_from_local(self, contact_pt):
        return np.arctan2(contact_pt[1], contact_pt[0])
        
    def get_normal_local(self, contact_angle):
        # any convex object can map 1-to-1 to a unit circle
        # this is in the local frame
        pass
    def get_tran_vel(self, contact_angle, vel):
        # get the vector from contact point to COM
        pt = self.get_contact_pt(contact_angle)
        pt = pt/np.linalg.norm(pt)
        return vel.dot(pt)*pt
    def get_rot_vel(self, contact_angle, vel):
        # get the vector from contact point to COM
        pt = self.get_contact_pt(contact_angle)
        norm = np.linalg.norm(pt)
        pt = pt/np.linalg.norm(pt)
        # linear_v = vel - vel.dot(norm)*norm
        linear_v = np.cross(vel, pt)  # |a||b|sin(theta)
        ang_v = linear_v / norm
        return ang_v
    def get_vel_in_local(self, vel):
        """
        given a velocity in the global frame, get the relative velocity in the local frame
        """
        tf_mat = self.tf_mat
        tf_mat = np.linalg.inv(tf_mat)
        return tf_mat[:2,:2].dot(vel)
    
    def get_vel_at_contact(self, contact_angle, tran_vel, rot_vel):
        # get the contact poitn velocity, assuming the object is undergoing velocity
        pass
    def get_cur_vel_at_contact(self, contact_angle):
        # get the contact point velocity based on the object current velocity
        pass
    """
    for getting the limit surface
    """
    def get_total_friction_at_pt(self, f, r):
        # TODO: consider coefficient and mass
        torque = np.cross(r, f)
        load = np.array([f[0], f[1], torque[2]])
        return load
    def get_total_friction_from_v(self, v):
        """
        v: [vx, vy, omega]
        """
        # get the local velocity of each point
        vx = np.zeros(self.geo.grid.shape) + v[0]
        vy = np.zeros(self.geo.grid.shape) + v[1]
        vx = vx - v[2] * self.geo.grid_y
        vy = vy + v[2] * self.geo.grid_x
        v_size = np.sqrt(vx ** 2 + vy ** 2)
        fx = vx / v_size
        fy = vy / v_size
        fw = self.geo.grid_x * fy - self.geo.grid_y * fx
        F = np.array([fx[self.geo.grid].sum(), fy[self.geo.grid].sum(), 
                      fw[self.geo.grid].sum()]) * self.geo.resols[0] * self.geo.resols[1]
        return F
    def construct_ls_db(self, n_samples=2500, ax=None):
        """
        sample velocities, and compute the (F,M) values. Then store them into a db
        Then we can:
        1. use convex hull of the sampled (F,M) to construct the limit surface
        2. use nearest neighbor to find velocity
        """
        points = np.random.normal(size=(n_samples,3))
        print(points.shape)
        points = points / np.linalg.norm(points, axis=1).reshape((-1,1))
        fs = []
        for i in range(len(points)):
            f = self.get_total_friction_from_v(points[i])
            fs.append(f)
        fs = np.array(fs)

        hull = ConvexHull(fs)

        # triple
        x = fs[:,0]
        y = fs[:,1]
        z = fs[:,2]

        # Triangulate parameter space to determine the triangles
        tri = mtri.Triangulation(x,y,triangles=hull.simplices)

        # Plot the surface.  The triangles in parameter space determine which x, y, z
        # points are connected by an edge.
        if ax is not None:
            ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
        
#         if ax is not None:
#             ax.plot_trisurf(fs[:,0], fs[:,1], fs[:,2], linewidth=0.2, antialiased=True)
        
#         # Plot defining corner points
#         if ax is not None:
#             ax.plot(fs[:,0], fs[:,1], fs[:,2], "ko")

#         # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
#         for s in hull.simplices:
#             s = np.append(s, s[0])  # Here we cycle back to the first coordinate
#             if ax is not None:
#                 ax.plot(fs[s, 0], fs[s, 1], fs[s, 2], "r-")            
        self.hull = hull
        return hull
    def get_ls_vel(self, f, plot_scale=0.1, ax=None):
        # https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python
        if self.hull is None:
            print('constructing hull...')
            self.construct_ls_db(ax=ax)
        # find the face that the force projects to
        eq=self.hull.equations.T
        V,b=eq[:-1].T,eq[-1]
        alpha=-b/np.dot(V,f)
        alpha[alpha<=0] = np.inf
        face_idx = np.argmin(alpha)
        proj_pt = alpha[face_idx]*f
        # TODO: plot this
        normal = self.hull.equations[face_idx,:-1]        
        normal = normal / np.linalg.norm(normal) * plot_scale
        if ax is not None:
            plt.plot([proj_pt[0],proj_pt[0]+normal[0]],
                     [proj_pt[1],proj_pt[1]+normal[1]],
                     [proj_pt[2],proj_pt[2]+normal[2]])
        print('projected point: ', proj_pt)
        print('normal: ', normal)
        return normal
    def get_ls_f(self, vel, ax=None):
        # find the force that corresponds to the normal direction vel
        # the one with the max inner product
        if self.hull is None:
            print('constructing hull...')
            self.construct_ls_db(ax=ax)
        print('here')
        vel = vel / np.linalg.norm(vel)
        idx = np.argmax(self.hull.points.dot(vel))
        return self.hull.points[idx]
    
    
# TODO: implement cylinder and block sliders. Write an animation for these
class CylinderSlider(Slider):
    def __init__(self, transform, shape):
        super().__init__(transform, shape)
        self.radius = shape  # shape is a float
        self.type = 'cylinder'
        mins = np.array([-self.radius, -self.radius])
        maxs = np.array([self.radius, self.radius])
        resols = np.array([0.01, 0.01])
        self.geo = CylinderShape(shape, mins, maxs, resols)

    def get_contact_pt(self, contact_angle):
        return self.radius*np.array([np.cos(contact_angle),np.sin(contact_angle)])
    def get_contact_pt_in_world(self, contact_angle):
        contact_pt = self.get_contact_pt(contact_angle)
        return self.tf_mat[:2,:2].dot(contact_pt) + self.tf_mat[:2,2]

    def get_normal_local(self, contact_angle):
        return -np.array([np.cos(contact_angle),np.sin(contact_angle)])
    def get_vel_at_contact(self, contact_angle, tran_vel, rot_vel):
        linear_v = rot_vel * self.radius
        linear_v = np.array([0,0,linear_v])
        # get the vector pointing from the COM to the contact point. This is useful to get the
        # linear velocity at the contact point
        contact_pt = self.get_contact_pt(contact_angle)
        contact_pt_vec = self.tf_mat[:2,:2].dot(contact_pt)
        contact_pt_vec = contact_pt_vec / np.linalg.norm(contact_pt_vec)
        contact_pt_vec = np.array([contact_pt_vec[0], contact_pt_vec[1], 0])
        linear_v = np.cross(linear_v,contact_pt_vec)[:2]
        return tran_vel + linear_v
    def get_cur_vel_at_contact(self, contact_angle):
        # get the contact point velocity based on the object current velocity
        return self.get_vel_at_contact(contact_angle, self.tran_vel, self.rot_vel)
    
    
class BlockSlider(Slider):
    def __init__(self, transform, shape):
        super().__init__(transform, shape)
        self.width = shape[0]
        self.height = shape[1]
        shape = np.array(shape)
        self.shape = np.array(shape[:2])
        self.type = 'block'
        mins = -shape/2
        maxs = shape/2
        resols = np.array([0.01, 0.01])
        self.geo = BlockShape(shape, mins, maxs, resols)
        
    def get_contact_pt(self, contact_angle):
        pt = np.array([np.cos(contact_angle),np.sin(contact_angle)])
        # map to rectangular
        ratio = np.array([self.width,self.height])/2/np.abs(pt)
        ratio = np.min(ratio)
        pt = pt * ratio
        return pt
    def get_contact_pt_in_world(self, contact_angle):
        contact_pt = self.get_contact_pt(contact_angle)
        return self.tf_mat[:2,:2].dot(contact_pt) + self.tf_mat[:2,2]
    def get_normal_local(self, contact_angle):
        # this is useful to project the velocity to ignore friction
        pt = self.get_contact_pt(contact_angle)
        if (pt[0] > -self.width/2) and (pt[0] < self.width/2):
            norm = -np.array([0,np.sign(pt[1])])
        if (pt[1] > -self.height/2) and (pt[1] < self.height/2):
            norm = -np.array([np.sign(pt[0]),0])
        return norm
    def get_vel_at_contact(self, contact_angle, tran_vel, rot_vel):
        pt = self.get_contact_pt(contact_angle)
        linear_v = rot_vel * np.linalg.norm(pt)
        linear_v = np.array([0,0,linear_v])
        # get the vector pointing from the COM to the contact point. This is useful to get the
        # linear velocity at the contact point
        contact_pt = self.get_contact_pt(contact_angle)
        contact_pt_vec = self.tf_mat[:2,:2].dot(contact_pt)
        contact_pt_vec = contact_pt_vec / np.linalg.norm(contact_pt_vec)
        contact_pt_vec = np.array([contact_pt_vec[0], contact_pt_vec[1], 0])
        linear_v = np.cross(linear_v,contact_pt_vec)[:2]
        return tran_vel + linear_v
    def get_cur_vel_at_contact(self, contact_angle):
        # get the contact point velocity based on the object current velocity
        return self.get_vel_at_contact(contact_angle, self.tran_vel, self.rot_vel)

# TODO: what about robot models? How to infer that?
# maybe we can randomly select a point on the robot arm geometry, and assuming convex shape. Then
# iteratively refine until the robot geometry roughly covers the point and is normal to that.
# This pushing motion will be noisy, so we need a controller to decide the velocity after a certain
# period for feedback control

class System:
    def __init__(self, slider: Slider, dt):
        self.slider = slider
        self.dt = dt
    def get_contact_point(self, pusher_pt):
        """
        at current slider transform, get the current contact point given the pusher point in the world frame
        """
        pass
    def update_slider_transform(self, s0, v0, v1):
        """
        v0: velocity before updating the velocity of the slider. (e.g. default is static)
        v1: velocity after update
        both v0 and v1 are of form [vx, vy, omega]
        """
        # we use a simple euler integration
        s1 = s0 + self.dt * v1
        self.slider.update_transform(s1)
        return s1
    
    def get_vel_in_local(self, vel):
        """
        given a velocity in the global frame, get the relative velocity in the local frame
        """
        tf_mat = self.slider.tf_mat
        tf_mat = np.linalg.inv(tf_mat)
        return tf_mat[:2,:2].dot(vel)

    def update_slider_rel(self, contact_angle, pusher_rel_vel):
        """
        given the contact angle, update the slider velocity and transform.
        The pusher_rel_velocity is in the frame of the slider
        NOTE: in quasi-static assumption, we basically assume that the contact velocity is equivalent
        to the pusher velocity in the slider after the pusher-slider friction effect
        NOTE: the pusher and slider may lose contact if we don't project one to the other. We need
        to update the slider by projecting it at the contact to the pusher afterward.
        """
        # frictionless: we ignore the tangent velocity to the boundary, and only take the normal velocity
        print('input pusher_rel_vel: ', pusher_rel_vel)
        norm = self.slider.get_normal_local(contact_angle)
#         print('norm: ', norm)
        projection = pusher_rel_vel.dot(norm)
        projection = max(projection, 0)  # ignore when the object is departing
        pusher_rel_vel = projection * norm
        print('contact_angle: ', contact_angle)
        print('norm: ', norm)
        print('projection: ', projection)
        print('pusher_rel_vel: ', pusher_rel_vel)
        
        # if the pusher velociy is not pointing inside the object, then ignore
        
    
        pusher_vel = self.slider.tf_mat[:2,:2].dot(pusher_rel_vel)
        pt = self.slider.get_contact_pt(contact_angle)
        # get pt in the global
        pt_global = self.slider.tf_mat[:2,:2].dot(pt) + self.slider.tf_mat[:2,2]
        
        # quasi-static: don't count previous velocities anymore
        friction = np.array(pusher_rel_vel)
        friction = np.array([friction[0], friction[1], 0])
        friction = friction / np.linalg.norm(friction)
        f = self.slider.get_total_friction_at_pt(friction, pt)
        vel_dir = self.slider.get_ls_vel(f)  # this is normalized velocity. Only a direction
        # match the velocity of the pusher: the contact point needs to have similiar velocity size
        vel_contact = self.slider.get_vel_at_contact(contact_angle, vel_dir[:2], vel_dir[2])
        vel_scale = np.linalg.norm(pusher_rel_vel) / np.linalg.norm(vel_contact)
        vel_slider = vel_dir * vel_scale
        vel_dx = pusher_rel_vel - vel_contact * vel_scale
        
        
        self.slider.tran_vel = [0,0]
        self.slider.rot_vel = 0
        tran_vel = self.slider.tran_vel + self.slider.tf_mat[:2,:2].dot(vel_slider[:2])
        rot_vel = self.slider.rot_vel + vel_slider[2]
        
        tran_vel += self.slider.tf_mat[:2,:2].dot(vel_dx)
        v0 = np.array([self.slider.tran_vel[0],self.slider.tran_vel[1],self.slider.rot_vel])
        v1 = np.array([tran_vel[0],tran_vel[1],rot_vel])
        
        print('slider velocity: ', vel_slider)
        print('pusher velocity: ', pusher_rel_vel)
        print('contact velocity: ', vel_contact)

        
        
        transform = self.update_slider_transform(self.slider.transform, v0, v1)

        new_pt = self.slider.get_contact_pt_in_world(contact_angle)
        pt_dx = new_pt - pt_global
        print('new contact point: ', new_pt)
#         print('slider vs pusher: ', pt_dx - pusher_vel * dt)
#         print('pusher travel: ', pusher_vel * dt)

        # project the contact point to the pusher new waypoint
#         new_pt = self.slider.get_contact_pt_in_world(contact_angle)
#         # get the translation of the contact point
        dx = pusher_vel * self.dt - (new_pt - pt_global)
        tran_vel += dx  # project the slider so that the pusher is at the boundary
        transform[:2] += dx
#         # TODO: add cone control afterward



        self.slider.update_transform(transform)
        self.slider.update_vel(tran_vel, rot_vel)

    def update_slider(self, contact_angle, pusher_vel_in_slider):
        """
        given the contact angle, update the slider velocity and transform.
        The pusher_vel_in_slider is in the frame of the slider
        NOTE: in quasi-static assumption, we basically assume that 
        the object becomes static and follows pusher velocity again
        """
        self.update_slider_rel(contact_angle, pusher_vel_in_slider)
        
    
