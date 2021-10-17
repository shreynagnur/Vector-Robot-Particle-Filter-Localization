#Shrey Nagnur
## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    pass

from skimage import color
import anki_vector
import skimage
from anki_vector.events import Events
from anki_vector import annotate, events
from anki_vector.util import degrees, distance_mm, distance_inches, speed_mmps, Pose
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image

from markers import detect, annotator_vector

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter_provided import *
from utils import *

global flag_odom_init, last_pose
global grid, gui, pf
#particle filter functionality

class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        print(m_confident)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
flag_odom_init = False
last_pose = anki_vector.util.Pose(0,0,0,angle_z=anki_vector.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (4.75, 8, 90)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)


def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a anki_vector.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Vector's camera.

    This can be called using the following:

    markers, camera_image = marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: anki_vector.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Vector's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Vector
    image_raw = robot.camera.latest_image.raw_image
    image = np.array(image_raw)

    # Convert the image to grayscale
    image = color.rgb2gray(image)
    
    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_raw.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator_vector.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator_vector.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


def run(): #--serial 003066c2

    global flag_odom_init, last_pose
    global grid, gui, pf
    args = anki_vector.util.parse_command_args()

    # Default Values of camera intrinsics matrix
    camera_settings = np.array([
        [296.54,      0, 160],    # fx   0  cx
        [     0, 296.54, 120],    #  0  fy  cy
        [     0,      0,   1]     #  0   0   1
    ], dtype=np.float)

    with anki_vector.Robot(serial=args.serial, show_viewer = False) as robot:


        # start streaming
        robot.camera.init_camera_feed()

        ################### 003066c2

        # YOUR CODE HERE
        if True:
            #robot.behavior.say_text('2')
            #time.sleep(2)
            robot.behavior.set_head_angle(degrees(2))
            robot.behavior.set_lift_height(0.0)

            reached_goal = False
            converged_final = False
            initial_speed = 7

            i = 0
            picked_up = False


            while not reached_goal:
                picked_up = robot.status.is_picked_up
                if not converged_final: #not converged
                    robot.motors.set_wheel_motors(initial_speed, -1 * initial_speed) #spin
                    pose = robot.pose #only reset this before computing odometry
                    odometry_info = compute_odometry(pose)
                    last_pose = pose #reset for next run

                    markers, camera_image = marker_processing(robot, camera_settings, show_diagnostic_image=True)
                    x, y, h, mean_good = pf.update(odometry_info, markers)
                    gui.show_particles(pf.particles)
                    gui.show_mean(x, y, h, mean_good)
                    gui.show_camera_image(camera_image)
                    gui.updated.set()
                    if mean_good:
                        robot.motors.stop_all_motors()
                        h -= 5 #overshoot due to spinning
                        converged_final = True
                        #print('FINAL CONVERGED POSE: ')
                        #print('x = ' + str(x) + ' inches')
                        #print('y = ' + str(y) + ' inches')
                        #print('h = ' + str(h) + ' degrees')
                        time.sleep(0.1)
                else: #converged, go to goal
                    robot.motors.stop_all_motors()
                    print('Going to goal now')
                    diff_x = goal[0] - x #diff in inches
                    diff_y = goal[1] - y #diff in inches
                    drive_margin = 2 #mm margin to slow to stop
                    degree_factor = 1 #degree margin to slow to stop
                    move_angle = math.atan2(diff_y, diff_x) #radians
                    move_angle = math.degrees(move_angle) #degrees
                    distance = grid_distance(goal[0], goal[1], x, y) * 25.4 #inches to mm

                    val1 = diff_heading_deg(move_angle, h) * degree_factor
                    robot.behavior.turn_in_place(degrees(val1), speed = degrees(20))

                    val2 = distance - drive_margin
                    robot.behavior.drive_straight(distance_mm(val2), speed_mmps(30)) #mm

                    val3 = diff_heading_deg(goal[2], move_angle) * degree_factor
                    robot.behavior.turn_in_place(degrees(val3), speed = degrees(20))


                    #should face right (short 2) after above
                    #6/26 = 0.23
                    #10/18 = 0.55
                    reached_goal = True
                    robot.motors.stop_all_motors()
                ###################
        robot.behavior.say_text('ready')
        robot.motors.stop_all_motors()
        robot.behavior.set_head_angle(degrees(5))
        time.sleep(2)
        pose = robot.pose
        #k = 1
        #flag = 1
        robot.world.connect_cube()
        while True:
            if (robot.world.connected_light_cube):
                #robot.behavior.say_text('working')
                robot.behavior.go_to_object(robot.world.connected_light_cube, distance_mm(70.0))
                robot.behavior.pickup_object(robot.world.connected_light_cube, num_retries = 5)
                break
            else:
                robot.behavior.look_around_in_place()
                robot.world.connect_cube()
        robot.motors.stop_all_motors()
        #robot.behavior.say_text('success')
        for i in range(3):
            if (i != 0):
                robot.behavior.say_text('ready')
                robot.motors.stop_all_motors()
                pose = robot.pose
                #k = 1
                #flag = 1
                robot.world.connect_cube()
                while True:
                    if (robot.world.connected_light_cube):
                        #robot.behavior.say_text('working')
                        robot.behavior.go_to_object(robot.world.connected_light_cube, distance_mm(70.0))
                        robot.behavior.pickup_object(robot.world.connected_light_cube, num_retries = 5)
                        break
                    else:
                        robot.behavior.look_around_in_place()
                        robot.world.connect_cube()
            robot.world.connect_cube()
            robot.behavior.go_to_pose(pose)
            #robot.behavior.say_text('delivery')
            turn('right', 180, robot)
            drive('short', robot)
            turn('left', 90, robot)
            drive('long', robot)
            turn('left', 90, robot)
            drive('shortplus', robot)
            robot.behavior.place_object_on_ground_here(num_retries = 3)
            if i < 2:
                turn('right', 180, robot)
                drive('shortplus', robot)
                turn('right', 90, robot)
                drive('long', robot)
                turn('right', 90, robot)
                drive('short', robot)
            else:
                robot.motors.stop_all_motors()
                robot.behavior.say_text('done')



def turn(direction, angle, robot):
    if direction == 'right':
        #robot.behavior.say_text('right ' + str(angle))
        angle *= -1
    elif direction == 'left':
        #robot.behavior.say_text('left ' + str(angle))
        pass
    else:
        print('what are you doing')
    robot.behavior.turn_in_place(degrees(angle))

def drive(distance, robot):
    if distance == 'short':
        #robot.behavior.say_text('short')
        val = 90
        speed = 50
    elif distance == 'shortplus':
        #robot.behavior.say_text('short plus')
        val = 175
        speed = 50
    elif distance == 'long':
        #robot.behavior.say_text('long')
        val = 425
        speed = 50
    else:
        print('what are you doing')
    robot.behavior.drive_straight(distance_mm(val), speed_mmps(speed))


class VectorThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        run()        

if __name__ == '__main__':

    #serial = '003066C2'

    # vector thread
    vector_thread = VectorThread()
    vector_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()