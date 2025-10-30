**Ubuntu 24.04 (Noble)**
- Download: https://cdimage.ubuntu.com/noble/daily-live/current/

**System prerequisites**
```
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential \
                    libglfw3 libglfw3-dev
```

**Install pyenv**
```
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
source ~/.bashrc
```

**Create and activate Python env**
```
pyenv install 3.11
pyenv virtualenv 3.11 g1env311
pyenv activate g1env311
python -V   # should be 3.11.x
```

**Clone cyclonedds**
```
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

**Clone Unitree SDK**
```
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git

# Install Unitree SDK Python bindings 
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```

**Set up simulation**
```
git clone https://github.com/unitreerobotics/unitree_mujoco.git
pip install mujoco pygame

# Replace config.py and unitree_sdk2py_bridge.py for unitree_mujoco
# Change to ENABLE_ELASTIC_BAND = False in config,py to allow ground contact
```

**Run simulation**
```
cd ~/unitree_mujoco/simulate_python
python unitree_mujoco.py
```