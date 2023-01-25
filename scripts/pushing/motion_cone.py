"""
define motion cones to use
"""

class SimpleMotionCone:
    def __init__(self, angle):
        self.angle = angle
    def get_angle(self, p):
        return self.angle