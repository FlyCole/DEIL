import pdb
import sys

sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rospy
import torch
import numpy as np

sys.path.insert(0, '/home/yifei/ws_moveit/devel/lib/python3/dist-packages')
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

import cv2
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class AGIRobot:
    def __init__(self):
        # print_green("Initializing AGI Robot...")
        rospy.init_node('agi_robot')

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize joint states subscriber
        self._joint_states = None
        self._moving = False
        rospy.Subscriber('/franka_state_controller/joint_states', JointState, self._joint_states_callback)

        # self._ik_solver = KinematicsSolver("/home/yifei/franka_ws/src/agi_control/assets/panda_arm.urdf", 1)

        # Initialize torq control publishers
        self._torque_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=10)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        # self._gripper_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)

        # Initialize gripper
        self.gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_moving = False
        self.gripper_client.wait_for_server()
        self.open_gripper()


        # self.keyboard = keyboard.Listener(on_press=self.on_press)
        # self.keyboard.start()

        # Initialize home position
        # self._home_q = torch.tensor([-0.12749246, -0.50128009,  0.10842198, -2.3844344, 0.05500573,  1.88400912, 0.73633193], device=self._device)
        # self._home_q = torch.tensor([-0.0887921194442648, 0.06434070869956818, 0.022036812649484268, -1.5586588080369375, 0.028370995455318057, 1.6465035621727444, 0.8045240546166895], device=self._device)
        self._home_q = torch.tensor([0.05674777057453208, 0.25735470228567087, -0.1774618511891659, -0.8739350225381682, 0.09381746787537433, 1.203769447353151, 0.751426609197221], device=self._device)
        rospy.sleep(1.0)  # OPT: remove, or somehow wait for everything to be ready.
        # print_green('Robot initialisation complete!')

    def _joint_states_callback(self, msg):
        """
        Callback function for joint states subscriber
        """
        curr_joint_states = np.array(msg.position)

        # Check if the robot is moving
        if self._moving and self._joint_states is not None:
            tol = 0.0001
            q_diff = np.abs(curr_joint_states - self._joint_states)
            q_matches = np.all(q_diff < tol)
            if q_matches:
                self._moving = False

        self._joint_states = curr_joint_states

    def _get_joint_states(self):
        """
        Get the current joint states
        """
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        return joint_states

    def recover_from_error(self):
        state_msg = rospy.wait_for_message(
            "/franka_state_controller/franka_states", FrankaState
        )
        if state_msg.robot_mode == FrankaState.ROBOT_MODE_MOVE:
            return True

        client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        client.wait_for_server()

        if client.get_state() == GoalStatus.SUCCEEDED:
            return True

        goal = ErrorRecoveryGoal()
        state = client.send_goal_and_wait(goal)
        succeeded = state == GoalStatus.SUCCEEDED

        if succeeded:
            rospy.loginfo("Recovered from error")
        else:
            rospy.logerr("Failed to recover from error")
        return succeeded

    def move_to_q(self, q, dur=10, blocking=False):
        if not self.recover_from_error():
            return

        if isinstance(q, torch.Tensor):
            q = q.cpu().numpy()

        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        # msg.header.stamp = rospy.Time(0)
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        point = JointTrajectoryPoint()
        point.positions = q
        point.time_from_start = rospy.Duration(dur)
        point.velocities = [0.0] * 7    # important to make sure velocity continuity
        # point.accelerations = [0.0] * 7  # important to make sure acceleration continuity
        msg.points = [point]
        self._torque_pub.publish(msg)
        self._moving = True

        if blocking:
            while self._moving:
                rospy.sleep(0.1)

    def move_to_goal_set(self, goal_set, blocking=True):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        goal_set = goal_set.cpu().numpy()
        total_time = 0.1
        for i in range(len(goal_set)-1):
            point = JointTrajectoryPoint()
            # Convert joint states to JointTrajectoryPoint message
            point.positions = goal_set[i+1]
            point.velocities = [0.01] * 7
            # point.accelerations = [0.0] * 7

            # Calculate the duration based on the joint difference
            joint_diff = np.abs(goal_set[i+1] - goal_set[i])
            joint_max_dur = np.max(joint_diff) / 0.3
            dur = np.max([joint_max_dur, 1])
            print(joint_max_dur)

            # Convert time to duration
            total_time += dur
            point.time_from_start = rospy.Duration(total_time)
            msg.points.append(point)
        pdb.set_trace()
        msg.header.stamp = rospy.Time.now()
        self._torque_pub.publish(msg)

    def move_to_goal_batch(self, start_q_batch, goal_pose_batch, blocking=True):
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        traj, traj_dt = self._ik_solver.motion_generation_batch(start_q_batch, goal_pose_batch)

        total_time = 0.1
        for i in range(traj.shape[0]):
            point = JointTrajectoryPoint()
            # Convert joint states to JointTrajectoryPoint message
            point.positions = traj[i].position.detach().cpu().numpy()
            point.velocities = traj[i].velocity.detach().cpu().numpy()
            point.accelerations = traj[i].acceleration.detach().cpu().numpy()
            # Convert time to duration
            total_time += traj_dt
            point.time_from_start = rospy.Duration(total_time)
            # point.time_from_start = rospy.Duration(i // 100 + time_offset, 10000000 * (i % 100))
            # point.time_from_start = rospy.Duration(i // 100 + time_offset, 10000000 * (i % 100))
            msg.points.append(point)
        # joint_angles = joint_angles[0][0].cpu().detach().numpy()
        joint_angles = traj[-1].position.detach().cpu().numpy()
        msg.header.stamp = rospy.Time.now()
        self._torque_pub.publish(msg)


    def go_home(self, dur=10):
        self.move_to_q(self._home_q, dur, blocking=True)

    def capture_rgbd(self):
        img_msg = rospy.wait_for_message('/camera/color/image_rect_color', Image)
        depth_msg = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width, -1)
        return img, depth

    def construct_pcd(self, img, depth):
        cam_msg = self._tf_buffer.lookup_transform('panda_link0', 'camera_color_optical_frame', rospy.Time(0))
        cam_pose = TransformStamped_to_array(cam_msg)
        from tf.transformations import quaternion_matrix
        T_WC = quaternion_matrix(cam_pose[3:])
        T_WC[:3, 3] = cam_pose[:3]
        T_CW = np.linalg.inv(T_WC)

        import open3d as o3d
        height = depth.shape[0]
        width = depth.shape[1]
        color_o3d = o3d.geometry.Image(img)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=0.8, convert_rgb_to_intensity=False)
        intrinsics = np.array([[635.7573852539062,  0.0,               632.9324951171875],
                               [0.0,        635.176513671875,         365.1086120605469],
                               [0.0,        0.0,               1.0]])
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsics, T_CW)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        vis.run()

    def close_gripper(self, blocking=False):
        # Close gripper until contact detected. Should be idempotent, so that closing again does nothing.
        real_joint_states = self._get_joint_states().position
        if np.all(real_joint_states[-2] < 0.002):
            return
        grasp_epsilon = GraspEpsilon(inner=0.04, outer=0.04)
        goal = GraspGoal(width=0.04, epsilon=grasp_epsilon, force=5, speed=0.05)
        self.gripper_moving = True
        self.gripper_client.send_goal(goal, done_cb=self.gripper_done_cb)
        if blocking:
            self.gripper_client.wait_for_result()

    def open_gripper(self, blocking=False):
        # Should be idempotent.
        real_joint_states = self._get_joint_states().position
        if np.all(real_joint_states[-2] > 0.075):
            return
        goal = GraspGoal(width=0.08, speed=0.05)
        self.gripper_moving = True
        self.gripper_client.send_goal(goal, done_cb=self.gripper_done_cb)
        if blocking:
            self.gripper_client.wait_for_result()

    def gripper_done_cb(self, status, result):
        self.gripper_moving = False

    def on_press(self, key):
        try:
            if key.char == 'o':
                print('Opening the gripper...')
                rospy.loginfo('Opening the gripper...')
                self.open_gripper(blocking=True)
            elif key.char == 'c':
                print('Closing the gripper...')
                rospy.loginfo('Closing the gripper...')
                self.close_gripper(blocking=True)
                input()
            elif key.char == 's':
                print('Saving joint states...')
                rospy.loginfo('Saving joint states...')
                self.record_js()
        except AttributeError:
            pass

    def record_js(self):
        # Record joint states
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        joint_states = np.array(joint_states.position[:7])
        js_str = f'{joint_states[0]} {joint_states[1]} {joint_states[2]} {joint_states[3]} {joint_states[4]} {joint_states[5]} {joint_states[6]} \n'
        print("Joint states: ", joint_states)
        with open(self.js_file, 'a') as f:
            f.write(js_str)
        return joint_states

    def get_nerf_traj(self):
        # Read joint states from file
        with open(self.js_file, 'r') as f:
            lines = f.readlines()
            joint_states = [list(map(float, line.strip().split())) for line in lines]

        # Convert to numpy array
        joint_states = np.array(joint_states)
        joint_states = torch.tensor(joint_states, device=self._device, dtype=torch.float32)

        # Map js to Goal poses
        goal_set = []
        for i in range(len(joint_states)-1):
            end_js = joint_states[i+1]
            goal_pose = self._ik_solver.forward_kinematics(end_js)
            goal_set.append(goal_pose)
        goal_set = torch.stack(goal_set).view(-1, 7).to(device=self._device, dtype=torch.float32)
        # joint_states = joint_states[:-1]

        return joint_states, goal_set


def get_traj(scene_center):
    scene_center = np.array([0.42, -0.03, 0.02117608])

    # Initialise 3d viewer
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=800, height=800)
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    # x-y plane sample
    ret_traj = []
    sample_bin_0 = [0, scene_center, 2 * np.pi, np.pi / 2.5, 0.55]
    sample_bin_1 = [0, scene_center, 2 * np.pi, np.pi / 3, 0.45]
    sample_bin_2 = [0, np.zeros(3), np.pi, np.pi / 4, 0.4]
    sample_bin_3 = [0, np.zeros(3), np.pi, np.pi / 3, 0.6]
    # sample_bin_3 = [0, scene_center, 2 * np.pi, np.pi / 2.5, 0.45]
    sample_bin = [sample_bin_0, sample_bin_1, sample_bin_2, sample_bin_3]

    for i in range(len(sample_bin)):
        start_theta = -np.pi / 2  # always starting from nearing the robot to avoid joint limit
        sample_num = sample_bin[i][0]
        sample_center = sample_bin[i][1]
        sample_range = sample_bin[i][2]
        sample_phi = sample_bin[i][3]
        sample_distance = sample_bin[i][4]
        for j in range(sample_num):
            sample_theta = start_theta + sample_range / sample_num * j

            sample_pos = sample_center + sample_distance * np.array([np.cos(sample_theta) * np.cos(sample_phi),
                                                                     np.sin(sample_theta) * np.cos(sample_phi),
                                                                     np.sin(sample_phi)])
            lookat_vec = scene_center - sample_pos
            sample_ori = look_at(sample_pos, scene_center)
            T = np.identity(4)
            T[:3, :3] = sample_ori
            T[:3, 3] = sample_pos

            # ret_traj.append([T, lookat_vec])
            ret_traj.append(T)
            viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(T))

    # Initial pose in local coordinate
    T_WO = np.eye(4)
    T_WO[:3, 3] = scene_center
    T_OC = np.eye(4)
    # T_OC[:3, 3] = [-0.25, -0.35, 0.0]
    T_OC[:3, 3] = [-0.25, -0.38, 0.03]
    # T_OC[:3, 3] = [-0.3, -0.32, 0.04]
    local_target = np.array([0.0, 0.0, 0.0])
    lookat_vec = local_target - T_OC[:3, 3]
    T_OC[:3, :3] = look_at(T_OC[:3, 3], local_target)[:3, :3]
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(T_WO @ T_OC))

    k = 12
    delta_x = R.from_euler("X", -np.pi / 15).as_matrix()
    # delta_z = R.from_euler("z", -np.pi / 10.1).as_matrix()
    delta_z = R.from_euler("z", -np.pi / 20).as_matrix()

    scan_lines_count = 7
    for j in range(scan_lines_count):
        for _ in range(k):
            T_OC[:3, 3] = delta_x @ T_OC[:3, 3]
            lookat_vec = local_target - T_OC[:3, 3]
            T_OC[:3, :3] = look_at(T_OC[:3, 3], local_target)[:3, :3]
            T_WC = T_WO @ T_OC
            viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(T_WC))
            # ret_traj.append([T_WC, T_WO[:3, :3] @ lookat_vec])
            ret_traj.append(T_WC)

        if j < scan_lines_count - 1:
            T_OC[:3, 3] = delta_z @ T_OC[:3, 3]
            lookat_vec = local_target - T_OC[:3, 3]
            T_OC[:3, :3] = look_at(T_OC[:3, 3], local_target)[:3, :3]
            T_WC = T_WO @ T_OC
            viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(T_WC))
            # ret_traj.append([T_WC, T_WO[:3, :3] @ lookat_vec])
            ret_traj.append(T_WC)

        delta_x = np.linalg.inv(delta_x)
        delta_z = np.linalg.inv(delta_z)

    viewer.run()
    viewer.destroy_window()

    return ret_traj


def look_at(curr, target):
    z = target - curr
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    look_at = np.eye(4)
    look_at[:3, 0] = x
    look_at[:3, 1] = y
    look_at[:3, 2] = z
    look_at[:3, 3] = curr

    return look_at


if __name__ == '__main__':
    agi_robot = AGIRobot()
    agi_robot.recover_from_error()

    # Test depth image
    # img, depth = agi_robot.capture_rgbd()
    # agi_robot.construct_pcd(img, depth)
