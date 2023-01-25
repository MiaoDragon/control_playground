"""
difference from v1:
now we store the limit surface as a knn tree.
we separate the storage of existing F as a size vector and a direction vector.
Later when we have a new F input, we find the nearest F, and use the corresponding
V
"""


import numpy as np
from scipy.spatial import ConvexHull, KDTree
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class LimitSurface:
    def __init__(self):
        pass
    def get_twist(self, F):
        """
        to find the velocity direction as the normal vector pointing outward
        """
        pass
    def get_load(self, V):
        """
        to find the friction force given the twist
        this finds the maximal point in the convex shape in the direction of V
        """
        pass
    def get_load_point_vel(self, p, v_p):
        """
        obtain the pushing force at point p, and the load F as a result of it
        twist could also be reported
        """        
        pass


# ## Limit surface modeling: through 2D grid to equip with model identification

# In[4]:


class GridLimitSurface(LimitSurface):
    def __init__(self, friction_coeff, pressure_dist, resols):
        """
        shape: 
        friction_coeff: NxM (width x height)
        pressure_dist: NxM
        """
        super().__init__()
        self.friction_coeff = friction_coeff
        self.pressure_dist = pressure_dist
        self.w = friction_coeff.shape[0] * resols[0]
        self.h = friction_coeff.shape[1] * resols[1]
        self.mat_x, self.mat_y = np.indices(self.friction_coeff.shape)
        self.mat_x = self.mat_x
        self.mat_y = self.mat_y
        self.Fkdtree = None
        self.Vkdtree = None
        self.fs_length = None

        self.com = np.array([self.mat_x.shape[0]/2, self.mat_x.shape[1]/2])
        self.resols = resols
    def get_load(self, twist):
        """
        given a twist in the shape of: Bx3, [x,y,omega]
        find corresponding loads in the shape of: Bx3
        Notice that the load is in the reverse direction of the friction, and it is
        in the direction of velocity at each point
        """
        # get the normal direction of the twist
        twist = twist / np.linalg.norm(twist, axis=1).reshape((-1,1))
        twist = twist.reshape((len(twist),1,1,3))
        mat_x = self.mat_x.reshape((1,len(self.mat_x),len(self.mat_x[0]),1)) - self.com[0]
        mat_y = self.mat_y.reshape((1,len(self.mat_y),len(self.mat_y[0]),1)) - self.com[1]
        mat_x = mat_x * (self.w/self.mat_x.shape[0])  # scale to width and height
        mat_y = mat_y * (self.h/self.mat_y.shape[0])
        # obtain the velocity at each point
        v = np.zeros((len(twist),len(self.mat_x),len(self.mat_x[0]),3))
        v = v + twist
        v[...,0] = v[...,0] - v[...,2] * mat_y[...,0]
        v[...,1] = v[...,1] + v[...,2] * mat_x[...,0]
        v[...,2] = 0
        v = v / np.linalg.norm(v, axis=3, keepdims=True)
        # the angular velocity is omitted
        
        friction_coeff = self.friction_coeff.reshape((1,len(self.friction_coeff),
                                                      len(self.friction_coeff[0]),1))
        pressure_dist = self.pressure_dist.reshape((1,len(self.pressure_dist),
                                                      len(self.pressure_dist[0]),1))

        load = v * friction_coeff * pressure_dist  # BxNxMx3
        load[...,2] = mat_x[...,0]*load[...,1] - mat_y[...,0]*load[...,0]
        load = load.mean(axis=(1,2))  # result shape: Bx3
        return load
    
    def construct_limit_surface(self, n_samples=2500, ax=None):
        twists = np.random.normal(size=(n_samples,3))
        twists = twists / np.linalg.norm(twists, axis=1, keepdims=True)
        fs = self.get_load(twists)
        self.fs_len = np.linalg.norm(fs, axis=1)
        fs = fs / self.fs_len.reshape((-1,1))
        self.Fkdtree = KDTree(fs, leafsize=20, copy_data=True)
        self.Vkdtree = KDTree(twists, leafsize=20, copy_data=True)

        print('Fkdtree shape: ', self.Fkdtree.data.shape)
        print('Vkdtree shape: ', self.Vkdtree.data.shape)

    def get_twist(self, F, plot_scale=0.1, ax=None):
        """
        F: size of B x 3
        """
        if self.Fkdtree is None:
            self.construct_limit_surface(ax=ax)

        k = 5  # an average of the neighborhood
        _, indices = self.Fkdtree.query(F/np.linalg.norm(F, axis=1, keepdims=True), k)  # indices: B x k
        # find the corresponding Vs
        Vs = self.Vkdtree.data[indices, :]  # B x k x 3
        Vs = np.mean(Vs, axis=1)  # B x 3
        Vs = Vs / np.linalg.norm(Vs, axis=1, keepdims=True)

        return Vs
    def get_load_point_vel(self, p, v_p, v_err=1e-3, f_err=5e-2):
        """
        obtain the pushing force at point p, and the load F as a result of it
        twist could also be reported
        """
        """
        # * step 1: find candidate loads such that
        # v_p = J_p Delta H(F)
        # * step 2: from the candidates, select ones that satisfy:
        # F = J^T f, i.e., F[2] = p[0]F[1] - p[1]F[0]
        # return the one with the least error. If no one is found, report failure
        """
        # * step 1: find candidate loads such that
        # v_p = J_p Delta H(F)
        V = np.array(self.Vkdtree.data)
        V = V / np.linalg.norm(V, axis=1).reshape((-1,1))  # Bx3
        # find Jp V = v
        v = np.zeros((len(V),2))
        v[:,0] = V[:,0] - V[:,2]*p[1]
        v[:,1] = V[:,1] + V[:,2]*p[0]
        v = v / np.linalg.norm(v, axis=1).reshape((-1,1))        
        # cosine similarity is the largest
        sim = v.dot(v_p)/np.linalg.norm(v_p)
        # check the largest ones
        # print('sorted similarity scores: ')
        # print(np.sort(sim)[::-1])
        cand_mask = sim > (1-v_err)
        selected_V = V[cand_mask, :]

        cand_loads = self.get_load(selected_V)
        
        # * step 2: from the candidates, select ones that satisfy:
        # F = J^T f, i.e., F[2] = p[0]F[1] - p[1]F[0]
        # error function we use here: (LHS-RHS)/norm(F)
        
        error = np.abs(p[0]*cand_loads[:,1] - p[1]*cand_loads[:,0] - cand_loads[:,2])
        error = error / np.linalg.norm(cand_loads, axis=1)
        # print('sorted errors: ')
        # print(np.sort(error))
        cand = error < f_err
        # check whether it is solved: i.e. cand has at least one
        if np.sum(cand) == 0:
            return None
        return cand_loads[np.argmin(error)]
    
# In[43]:


import matplotlib.patches as matpat

def visualize(p, v_pusher, fl, fu, H, F, f):
    alpha = H.friction_coeff*H.pressure_dist
    alpha = alpha / alpha.max()    
    xmin = -H.w/2*1.5
    xmax = H.w/2 * 1.5
    ymin = -H.h/2*1.5
    ymax = H.h/2*1.5
    fig, ax = plt.subplots()
    plt.axis([xmin, xmax, ymin, ymax])        
    # draw the grid
    dx = H.w / H.mat_x.shape[0]
    dy = H.h / H.mat_x.shape[1]
    for i in range(H.mat_x.shape[0]):
        for j in range(H.mat_x.shape[1]):
            x = -H.w/2 + i*dx
            y = -H.h/2 + j*dy
            slider_ij = plt.Rectangle((x,y), dx, dy, color='gray', alpha=alpha[i,j])
            # we approxiate each rectangular grid by a circle to be symmetric
            
            # TODO: adjust color based on value
            ax.add_patch(slider_ij)

    # fl, fu
    fl = dx*2*fl/np.linalg.norm(fl)
    fu = dx*2*fu/np.linalg.norm(fu)
    f = dx*2*f/np.linalg.norm(f)

    fl_arrow = plt.arrow(p[0],p[1], fl[0],fl[1], width=dx/10,color='black')
    fu_arrow = plt.arrow(p[0],p[1], fu[0],fu[1], width=dx/10,color='black')
    f_arrow = plt.arrow(p[0],p[1], f[0],f[1], width=dx/10,color='blue')
    ax.add_patch(fl_arrow)
    ax.add_patch(fu_arrow)
    ax.add_patch(f_arrow)

    plt.annotate('f',
             xy=(p[0]+f[0]/2, p[1]+f[1]/2),
             color='blue')


    # velocities
    V = H.get_twist(F)
    # get slider velocity
    v = np.zeros((2))
    v[0] = V[0] - p[1] * V[2]
    v[1] = V[1] + p[0] * V[2]
    v = v / np.linalg.norm(v) *(dx*2)
    v_arrow = plt.arrow(p[0],p[1], v[0], v[1], width=dx/10,color='red')
    ax.add_patch(v_arrow)
    plt.annotate('v',
             xy=(p[0]+v[0]/2, p[1]+v[1]/2),
             color='red')

    vp = v_pusher / np.linalg.norm(v_pusher) *(dx*2)

    vp_arrow = plt.arrow(p[0],p[1], vp[0], vp[1], width=dx/10,color='green')
    ax.add_patch(vp_arrow)
    plt.annotate('v_pusher',
             xy=(p[0]+vp[0]/2, p[1]+vp[1]/2),
             color='green')


    fig.show()        

def point_pusher_model(p, v_pusher, fl, fu, H: LimitSurface, vis=None):
    """
    given the pushing point p, the velocity of the pusher v_pusher, and the limit surface model H.
    determine the velocity of the slider. (other intermediate values such as frictions too)
    NOTE: this should not fail, because it could only be one of three cases:
    - sticking
    - fl
    - fu
    """
    
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
            if vis:
                visualize(p, v_pusher, fl, fu, H, F, f)
            return F
    
    # * otherwise it's in the sliding mode
    # f = fl or fu
    F = np.array([fl[0], fl[1], p[0]*fl[1]-p[1]*fl[0]])
    Fl = np.array(F)
    V = H.get_twist(F)
    # verify the difference
    v = np.zeros((2))
    v[0] = V[0] - p[1]*V[2]
    v[1] = V[1] + p[0]*V[2]
    # check if it's fit
    normal = (fl+fu)/2
    flt = fl - fl.dot(normal)*normal
    vt = v - v.dot(normal)*normal  # get the tangent velocity
    if vt.dot(flt) >= 0:
        visualize(p, v_pusher, fl, fu, H, F, fl)
        return Fl
    else:
        F = np.array([fu[0], fu[1], p[0]*fu[1]-p[1]*fu[0]])
        Fu = np.array(F)
        visualize(p, v_pusher, fl, fu, H, F, fu)
        return Fu
