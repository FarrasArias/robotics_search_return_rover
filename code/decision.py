import numpy as np
import time

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function

frame_counter = 0
pos_list = []
yaw_list = []
rover_stuck = False
new_turn = -15

def decision_step(Rover):
    global first_position, frame_counter, pos_list, rover_stuck, yaw_list, new_turn

    #chose opposite side of steer angle for when the rover stops
    steer_right = -15
    steer_left = 15    
    if np.abs(Rover.steer) > 0 and Rover.vel > 1: 
        if Rover.steer > 0:
            new_turn = steer_right
        elif Rover.steer < 0:
            new_turn = steer_left

    #append the position of the rover through time to compare positions 
    frame_counter += 1
    frame_div = frame_counter%116
    if frame_div == 31:
        yaw_list.append(Rover.yaw)
    if frame_div == 115:
        pos_list.append(Rover.pos)
    if len(pos_list) > 2:
        pos_list.pop(0)
    
    #if positions are the same through time, it means the rover is stuck
    if (len(pos_list) >= 2 and 
        (pos_list[-1][0] < pos_list[0][0]+0.4 and pos_list[-1][0] > pos_list[0][0]-0.4) and
        (pos_list[-1][1] < pos_list[0][1]+0.4 and pos_list[-1][1] > pos_list[0][1]-0.4) and
        not Rover.near_sample):
        print("Rover is stuck!")
        rover_stuck = True
    
    #if it's stuck, turn the rover 30 degrees and clear all lists to check if it's stuck again
    if rover_stuck:
        print("Rover is Steering")
        Rover.throttle = 0
        # Release the brake to allow turning
        Rover.brake = 0
        Rover.steer = new_turn
        if len(yaw_list) > 0:
            if (yaw_list[-1] - yaw_list[0]) > 30 or (yaw_list[-1] - yaw_list[0]) < -30:
                yaw_list = []
                pos_list.pop(0)
                Rover.steer = 0
                rover_stuck = False
        
        
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!
    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None and not rover_stuck:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if (len(Rover.nav_angles) >= Rover.stop_forward) and not Rover.near_sample:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                print("steer",Rover.steer)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif (len(Rover.nav_angles) < Rover.stop_forward) and not Rover.near_sample:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            elif Rover.near_sample:
                Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward and not Rover.near_sample:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = new_turn # Could be more clever here about which way to turn
                elif len(Rover.nav_angles) < Rover.go_forward and Rover.near_sample:
                    Rover.steer = 0
                    Rover.brake = Rover.brake_set
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    elif not rover_stuck:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

