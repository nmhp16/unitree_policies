# ~/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py

import sys, struct, time
import numpy as np
import pygame
import mujoco

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.utils.thread import RecurrentThread

# High-state + wireless (go-family message types are fine for status/controller)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__SportModeState_      as SportModeState_default,
    unitree_go_msg_dds__WirelessController_  as WirelessController_default,
)

import config

# Select low-level IDL by robot family
if config.ROBOT == "g1":
    # G1/H1 use the HG family
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
    IDL_SLOTS = 35
else:
    # Go family
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
    IDL_SLOTS = 20

# DDS topics
TOPIC_LOWCMD              = "rt/lowcmd"
TOPIC_LOWSTATE            = "rt/lowstate"
TOPIC_HIGHSTATE           = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3  # (q, dq, tau_est) per actuator in sensordata


class UnitreeSdk2Bridge:
    """DDS <-> MuJoCo bridge (no default motion). Applies LowCmd to model; publishes states."""

    def __init__(self, mj_model, mj_data):
        # telemetry / watchdog
        self._rx_count        = 0
        self._last_print      = 0.0
        self._last_ctrl_norm  = 0.0
        self._last_rx_time    = 0.0
        self._watchdog_sec    = 0.2   # zero controls if no LowCmd within this window

        self.mj_model = mj_model
        self.mj_data  = mj_data

        # sizes
        self.num_motor         = int(self.mj_model.nu)     # MuJoCo actuators
        self.dim_motor_sensor  = MOTOR_SENSOR_NUM * self.num_motor
        self.dt                = float(self.mj_model.opt.timestep)

        # optional sensors
        self.have_imu_         = False
        self.have_frame_sensor_= False
        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name == "imu_quat":
                self.have_imu_ = True
            if name == "frame_pos":
                self.have_frame_sensor_ = True

        # ---------- actuator mapping ----------
        # Map DDS motor slots into MuJoCo ctrl indices. Be robust to scenes with fewer actuators.
        # For your G1 scene showing nu=29, we map range(min(nu, IDL_SLOTS)) -> indices [0..nu-1].
        self.map_idx = list(range(min(self.num_motor, IDL_SLOTS)))

        print(f"[Bridge] MuJoCo actuators (nu)={self.num_motor}, IDL slots={IDL_SLOTS}, mapped={len(self.map_idx)}")

        # ---------- DDS pubs/subs ----------
        # LowState publisher (feedback)
        self.low_state        = LowState_default()
        self.low_state_puber  = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread   = RecurrentThread(interval=self.dt, target=self.PublishLowState, name="sim_lowstate")
        self.lowStateThread.Start()

        # High-level pose/vel (for convenience UIs)
        self.high_state       = SportModeState_default()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread  = RecurrentThread(interval=self.dt, target=self.PublishHighState, name="sim_highstate")
        self.HighStateThread.Start()

        # Wireless controller publisher (created but NOT started until a joystick is set up)
        self.wireless_controller       = WirelessController_default()
        self.wireless_controller_puber = ChannelPublisher(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread  = None  # started in SetupJoystick() only

        # LowCmd subscriber
        print("[SIM] Subscribing topic=rt/lowcmd type=", LowCmd_.__module__ + "." + LowCmd_.__name__)
        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        # joystick plumbing
        self.joystick = None
        self.key_map  = {
            "R1":0,"L1":1,"start":2,"select":3,"R2":4,"L2":5,"F1":6,"F2":7,
            "A":8,"B":9,"X":10,"Y":11,"up":12,"right":13,"down":14,"left":15
        }

    # --------------- DDS → MuJoCo ----------------
    def LowCmdHandler(self, msg: LowCmd_):
        """Apply LowCmd motor commands to MuJoCo controls with clamping."""
        if self.mj_data is None:
            return

        M = min(len(self.map_idx), len(msg.motor_cmd))
        ctrl_abs = 0.0

        for dst_i in range(M):
            src_i = self.map_idx[dst_i]
            mc    = msg.motor_cmd[dst_i]

            # PD + feedforward torque
            val = (
                mc.tau
                + mc.kp * (mc.q  - self.mj_data.sensordata[src_i])
                + mc.kd * (mc.dq - self.mj_data.sensordata[src_i + self.num_motor])
            )

            # clamp to actuator range if limited
            if self.mj_model.actuator_ctrllimited[src_i]:
                lo, hi = self.mj_model.actuator_ctrlrange[src_i]
                if val < lo: val = lo
                if val > hi: val = hi

            self.mj_data.ctrl[src_i] = val
            ctrl_abs += abs(val)

        # telemetry + watchdog tick
        self._rx_count       += 1
        self._last_ctrl_norm += ctrl_abs
        self._last_rx_time    = time.time()

    # --------------- MuJoCo → DDS ----------------
    def PublishLowState(self):
        """Publish motor q/dq/tau_est and IMU if present. Enforce watchdog zeroing."""
        if self.mj_data is None:
            return

        # Watchdog: zero all controls if no LowCmd recently
        now = time.time()
        if (now - self._last_rx_time) > self._watchdog_sec:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = 0.0

        # Motor feedback
        M = min(len(self.map_idx), len(self.low_state.motor_state))
        for dst_i in range(M):
            src_i = self.map_idx[dst_i]
            self.low_state.motor_state[dst_i].q       = self.mj_data.sensordata[src_i]
            self.low_state.motor_state[dst_i].dq      = self.mj_data.sensordata[src_i + self.num_motor]
            self.low_state.motor_state[dst_i].tau_est = self.mj_data.sensordata[src_i + 2 * self.num_motor]

        # IMU/frame sensors
        if self.have_frame_sensor_:
            base = self.dim_motor_sensor
            if len(self.mj_data.sensordata) >= base + 16:
                self.low_state.imu_state.quaternion[:]   = self.mj_data.sensordata[base + 0: base + 4]
                self.low_state.imu_state.gyroscope[:]    = self.mj_data.sensordata[base + 4: base + 7]
                self.low_state.imu_state.accelerometer[:] = self.mj_data.sensordata[base + 7: base + 10]

        self.low_state_puber.Write(self.low_state)

        # debug rate print
        if now - self._last_print > 1.0:
            print(f"[Bridge] rx≈{self._rx_count:5d}/s  ctrl_abs_sum≈{self._last_ctrl_norm:.3f}")
            self._rx_count = 0
            self._last_ctrl_norm = 0.0
            self._last_print = now

    def PublishHighState(self):
        """Publish simple position/velocity for convenience UIs."""
        if self.mj_data is not None:
            base = self.dim_motor_sensor
            if len(self.mj_data.sensordata) >= base + 16:
                self.high_state.position[:] = self.mj_data.sensordata[base + 10: base + 13]
                self.high_state.velocity[:] = self.mj_data.sensordata[base + 13: base + 16]
        self.high_state_puber.Write(self.high_state)

    # --------------- Wireless controller (optional) ---------------
    def PublishWirelessController(self):
        if self.joystick is None:
            return

        pygame.event.get()
        key_state = [0] * 16
        key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
        key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
        key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
        key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
        key_state[self.key_map["R2"]] = (self.joystick.get_axis(self.axis_id["RT"]) > 0)
        key_state[self.key_map["L2"]] = (self.joystick.get_axis(self.axis_id["LT"]) > 0)
        key_state[self.key_map["A"]]  = self.joystick.get_button(self.button_id["A"])
        key_state[self.key_map["B"]]  = self.joystick.get_button(self.button_id["B"])
        key_state[self.key_map["X"]]  = self.joystick.get_button(self.button_id["X"])
        key_state[self.key_map["Y"]]  = self.joystick.get_button(self.button_id["Y"])
        key_state[self.key_map["up"]]    = self.joystick.get_hat(0)[1] > 0
        key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
        key_state[self.key_map["down"]]  = self.joystick.get_hat(0)[1] < 0
        key_state[self.key_map["left"]]  = self.joystick.get_hat(0)[0] < 0

        key_value = 0
        for i in range(16):
            key_value |= (int(bool(key_state[i])) << i)

        self.wireless_controller.keys = key_value
        self.wireless_controller.lx   = self.joystick.get_axis(self.axis_id["LX"])
        self.wireless_controller.ly   = -self.joystick.get_axis(self.axis_id["LY"])
        self.wireless_controller.rx   = self.joystick.get_axis(self.axis_id["RX"])
        self.wireless_controller.ry   = -self.joystick.get_axis(self.axis_id["RY"])

        self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        """Enable wireless-controller publisher only if a joystick is present."""
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected.")
            return

        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()

        if js_type == "xbox":
            self.axis_id   = {"LX":0,"LY":1,"RX":3,"RY":4,"LT":2,"RT":5,"DX":6,"DY":7}
            self.button_id = {"X":2,"Y":3,"B":1,"A":0,"LB":4,"RB":5,"SELECT":6,"START":7}
        elif js_type == "switch":
            self.axis_id   = {"LX":0,"LY":1,"RX":2,"RY":3,"LT":5,"RT":4,"DX":6,"DY":7}
            self.button_id = {"X":3,"Y":4,"B":1,"A":0,"LB":6,"RB":7,"SELECT":10,"START":11}
        else:
            print("Unsupported gamepad.")

        # Start wireless thread now that a joystick exists
        if self.WirelessControllerThread is None:
            self.WirelessControllerThread = RecurrentThread(
                interval=0.01,
                target=self.PublishWirelessController,
                name="sim_wireless_controller",
            )
            self.WirelessControllerThread.Start()

    # -------- debug helpers --------
    def PrintSceneInformation(self):
        print("\n<<------------- Link ------------->> ")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print("\n<<------------- Joint ------------->> ")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print("\n<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print("actuator_index:", i, ", name:", name)
        print("\n<<------------- Sensor ------------->>")
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                print("sensor_index:", index, ", name:", name, ", dim:", self.mj_model.sensor_dim[i])
            index += self.mj_model.sensor_dim[i]


class ElasticBand:
    def __init__(self):
        self.stiffness = 200
        self.damping   = 100
        self.point     = np.array([0, 0, 3])
        self.length    = 0
        self.enable    = True

    def Advance(self, x, dx):
        delta = self.point - x
        dist  = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(3)
        direction = delta / dist
        v         = np.dot(dx, direction)
        f         = (self.stiffness * (dist - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7: self.length -= 0.1
        if key == glfw.KEY_8: self.length += 0.1
        if key == glfw.KEY_9: self.enable  = not self.enable