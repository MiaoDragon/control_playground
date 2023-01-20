"""
NOTE: different from the base in:
1) we will get the mapping from (v,w) -> (F,I) -> r
2) to do this, we solve the optimization problem to find
    argmin |r1f2-r2f1-I|  for (r1,r2) in O
    where O stands for the object boundary
    we can do this optimization by sampling (r1,r2) in O, and then find the min one (or by threshold)
    It may be that min |r1f2-r2f1-I| > 0. In this case, there is no solution to the optimization problem
"""

from velocity_point_push import *


def load_to_position(slider: Slider, load):
    """
    load: [f1,f2,I]
    """
    # * step 1: sample points in the object geometry
    contact_angles = np.linspace(start=0, stop=np.pi*2, num=100, endpoint=False)
    contact_pts = slider.get_contact_pt(contact_angles)  # 2xN
    contact_pts = contact_pts.T

    # * step 2: find the one that minimizes |r1f2-r2f1-I|
    value = contact_pts[:,0]*load[1] - contact_pts[:,1]*load[0] - load[2]
    value = np.abs(value)
    small = 5*np.pi/180  # this is the wrench error we allow. Could have multiple solutions
    filter = value < small
    
    # * step 3: check if the force at the contact is pointing inside the object
