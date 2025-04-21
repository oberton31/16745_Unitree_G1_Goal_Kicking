import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("../unitree_robots/g1/scene_29dof.xml")
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

Kp = 200 
Kd = 10   

# target standing posture (actuator names taken fromn XML file)
standing_pose = {
    "left_hip_pitch": -0.2,
    "right_hip_pitch": -0.2,
    "left_knee": 0.4,
    "right_knee": 0.4,
    "left_ankle_pitch": -0.2,
    "right_ankle_pitch": -0.2,
    "waist_pitch": 0.1,
    "waist_roll": 0.0,
    "waist_yaw": 0.0,
}

def compute_gravity_compensation():
    mujoco.mj_forward(model, data)  
    return np.copy(data.qfrc_bias)  

def apply_pd_control():

    gravity_torques = compute_gravity_compensation() 
    for actuator_name, target_angle in standing_pose.items():
        try:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            joint_id = model.actuator_trnid[actuator_id][0]

            current_angle = data.qpos[model.jnt_qposadr[joint_id]]
            velocity = data.qvel[model.jnt_dofadr[joint_id]]

            torque = Kp * (target_angle - current_angle) - Kd * velocity + gravity_torques[joint_id]

            data.ctrl[actuator_id] = torque

        except Exception as e:
            print(f"Error processing {actuator_name}: {e}")

while viewer.is_running():
    apply_pd_control() 
    mujoco.mj_step(model, data)  
    viewer.sync()

