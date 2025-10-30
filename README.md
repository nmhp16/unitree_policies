sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential \
                    libglfw3 libglfw3-dev

python3 -m venv ~/g1env311
source ~/g1env311/bin/activate
python -V   # should be 3.11.x

cd ~
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git

# install Unitree SDK Python bindings (editable)
pip install -e ~/unitree_sdk2_python

# sim deps
pip install mujoco pygame

export CYCLONEDDS_NETWORK_INTERFACE=lo
export CYCLONEDDS_URI='<CycloneDDS><Domain id="any"><General><AllowMulticast>false</AllowMulticast></General><Network><Interfaces><Interface><Address>lo</Address></Interface></Interfaces></Network></Domain></CycloneDDS>'


cd ~/unitree_mujoco/simulate_python
python unitree_mujoco.py