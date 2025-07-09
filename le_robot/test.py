from robots.so100_follower import SO100Follower, SO100FollowerConfig
import time

cfg = SO100FollowerConfig(
    port="/dev/tty.usbmodem59700731401",
    use_degrees=True,
    max_relative_target=None,
    #max_relative_target=20.0,
)

robot = SO100Follower(cfg)
robot.connect()
obs = robot.get_observation()
print(obs)

# Test: Öffne Greifer ganz
#robot.send_action({"gripper.pos": 50})
#time.sleep(0.5) 
#robot.send_action({"wrist_flex.pos":0 }) 
#robot.send_action({"elbow_flex.pos":0})
#robot.send_action({"shoulder_lift.pos": 0.0})
robot.send_action({"shoulder_pan.pos": 100})
time.sleep(1)   
robot.send_action({"shoulder_pan.pos": -100})
time.sleep(0.5)  

robot.disconnect()


# Example action dictionary for the SO100Follower robot
#action = {
#    "shoulder_pan.pos":   0,   # ° relativ zur Home-Position
#    "shoulder_lift.pos": -76.79120879120879,
#    "elbow_flex.pos":     73.05494505494505,
#    "wrist_flex.pos":      44.92307692307692,
#    "wrist_roll.pos":      58.37362637362638,
#    "gripper.pos":        15,   # 0–100 % Öffnungs­weite
#}

#shoulder_pan.pos_min = -116
#shoulder_pan.pos_max = 116
#----
#shoulder_lift.pos_min: -102.68131868131869
#shoulder_lift.pos_max: 105
#---
#elbow_flex.pos_min: -100.7032967032967
#elbow_flex.pos_max: 92.87912087912088
#---
#wrist_flex.pos_min: -103.25274725274726
#wrist_flex.pos_max: 102.81318681318682
#---
#wrist_roll.pos_min: -165.93406593406593
#wrist_roll.pos_max: 167.6043956043956
#---    
#gripper.pos_min: 0
#gripper.pos_max: 100