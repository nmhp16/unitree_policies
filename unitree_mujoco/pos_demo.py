# pos_demo_g1.py  — G1 (unitree_hg) publisher using default IDL factories

import time, math
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize

# Raw IDL classes (for type registration)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, MotorCmd_

# "default" factories set correct field shapes (fixed-length arrays like `reserve`)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_   as LowCmd_default,
    unitree_hg_msg_dds__MotorCmd_ as MotorCmd_default,
)

import config

TOPIC = "rt/lowcmd"
N = 35   # G1/HG motor slots

def make_motor_cmd():
    m = MotorCmd_default()   # correct reserve shape pre-initialized
    m.mode = 0
    m.q    = 0.0
    m.dq   = 0.0
    m.tau  = 0.0
    m.kp   = 40.0
    m.kd   = 1.0
    # m.reserve left as default (correct fixed-size array)
    return m

def main():
    # Match the simulator's domain/interface
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)

    # Publisher must be created with the raw IDL class (not the default alias)
    pub = ChannelPublisher(TOPIC, LowCmd_); pub.Init()

    # Build a message with defaults (correct array lengths)
    cmd = LowCmd_default()
    cmd.mode_pr = 0
    cmd.mode_machine = 0
    # fill motor_cmd with N items
    cmd.motor_cmd = [make_motor_cmd() for _ in range(N)]
    # cmd.reserve left as default (correct fixed-size array)
    cmd.crc = 0

    rate = 200.0
    dt   = 1.0 / rate
    t0   = time.time()

    wiggle = {3, 9, 12, 15, 29}  # a few joints to move
    amp, freq = 0.30, 0.5

    try:
        while True:
            s = amp * math.sin(2 * math.pi * freq * (time.time() - t0))
            for i in range(N):
                m = cmd.motor_cmd[i]
                m.q  = s if i in wiggle else 0.0
                m.dq = 0.0
                m.tau= 0.0
            pub.Write(cmd)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()