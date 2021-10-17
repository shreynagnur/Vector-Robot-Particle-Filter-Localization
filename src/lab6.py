import asyncio
import concurrent
import anki_vector
from anki_vector.events import Events
from anki_vector import annotate
from anki_vector.util import degrees, distance_mm, speed_mmps
import time
from itertools import chain
import sys
import datetime
from imgclassification_sol import ImageClassifier as sol
import numpy as np
import joblib
from skimage import io

def whatever():
    trained = joblib.load('trained_model.pkl')
    with anki_vector.Robot(show_viewer=True) as robot:
        raw = robot.camera.latest_image
        classifier = sol()
        annotated = raw.annotate_image(scale = (2/3)) #426x240
        formatted = np.array(annotated)
        sliced = formatted[:, 53:373, :]
        feature_data = classifier.extract_image_features(np.array([sliced]))
        newState = trained.predict(feature_data)
        robot.camera.init_camera_feed()
        robot.motors.set_wheel_motors(0, 0)
        robot.behavior.set_head_angle(degrees(10))
        while (newState != 'hands'):
            robot.motors.set_wheel_motors(-10, 10)
        robot.motors.stop_all_motors()
        robot.behavior.drive_straight(distance_mm(100), speed_mmps(10))
        robot.behavior.turn_in_place(degrees(-50))
        robot.behavior.say_text('ready')

def main():
    for _ in range(4):
        robot.world.connect_cube()
        if robot.world.connected_light_cube:
            robot.behavior.pickup_object(robot.world.connected_light_cube, num_retries = 5)
        robot.behavior.turn_in_place(degrees(-180))
        robot.behavior.drive_straight(distance_mm(150), speed_mmps(20))
        robot.behavior.turn_in_place(degrees(90))
        robot.behavior.drive_straight(distance_mm(444), speed_mmps(20))
        robot.behavior.turn_in_place(degrees(90))
        robot.behavior.drive_straight(distance_mm(150), speed_mmps(20))
        robot.behavior.place_object_on_ground_here(num_retries = 3)
        robot.behavior.turn_in_place(degrees(-180))
        robot.behavior.drive_straight(distance_mm(150), speed_mmps(20))
        robot.behavior.turn_in_place(degrees(-90))
        robot.behavior.drive_straight(distance_mm(444), speed_mmps(20))
        robot.behavior.turn_in_place(degrees(-90))
        robot.behavior.drive_straight(distance_mm(150), speed_mmps(20))



if __name__ == "__main__":
    main()