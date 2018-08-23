#invite cool guys to the party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# boundaries
xlim=[0, 30]
ylim=[0, 20]

# set up initial configuration of plot
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
plt.xticks([])
plt.yticks([])
ln, = plt.plot([], [], 'ro', animated=True)

# global parameters
g = 9.8 # gravity
ag = np.array([0.0, -g]) # gravity vector
delta_t = 0.05
cor = 0.95 # damping factor at each accident with walls or other balls
friction_coeff = 0.97 # coefficient of friction at bottom edge along x direction
balls = []

class Ball():
    def __init__(self, radius, center, velocity):
        self.radius = 0.5 # it is different than that the argument "radius"
        self.position = center
        self.velocity = velocity
        self.scatter, = ax.plot([], [], 'o', markersize=radius)
        balls.append(self)
        self.ball_number = len(balls)

       
    def update(self):
        self.velocity += ag*delta_t
        self.position += self.velocity*delta_t

        if self.position[1] < ylim[0]: # hit bottom edge
            self.velocity[1] = - cor * self.velocity[1]
        if self.position[1] > ylim[1]: # hit top edge
            self.velocity[1] = - cor * self.velocity[1]
        if self.position[0] > xlim[1]: # hit right edge
            self.velocity[0] = - cor * self.velocity[0]
        if self.position[0] < xlim[0]: # hit left edge
            self.velocity[0] = - cor * self.velocity[0]

        # if ball is stuck at bottom edge apply friction
        if (self.position[1] < 0.2) and (self.velocity[1] < 1):
            self.velocity[0] = friction_coeff * self.velocity[0]
            
        
        # check if balls hit each other
        for ball in balls:
            if (ball.ball_number != self.ball_number)and (np.sqrt((ball.position[0]-self.position[0])**2 + (ball.position[1]-self.position[1])**2) < (ball.radius + self.radius)):
                self.velocity[0] = - cor * self.velocity[0]

        # clip position to make sure ball is within the boundaries
        self.position[0] = np.clip(self.position[0], xlim[0], xlim[1])
        self.position[1] = np.clip(self.position[1], ylim[0], ylim[1])
        
        self.scatter.set_data(self.position)
        
b1 = Ball(15.0, np.array([3., 18.]), np.array([1., 0.]))
b2 = Ball(9., np.array([28., 15.]), np.array([1.5, -20.]))
b3 = Ball(12., [5., 5.], [5., 5.])
b4 = Ball(14., [5., 15.], [5., -5.])


def init():
    return []

def animate(t):
    for ball in balls:
        ball.update()
    return [ball.scatter for ball in balls]
        
ani = animation.FuncAnimation(fig, animate, np.arange(0,100,delta_t), init_func=init, interval=10, blit=True)

plt.show()
