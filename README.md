# Kino-SMF: Kinodynamic Social Multilayer Feedback Navigation

Kino-SMF is a ROS 2 navigation stack for robots that move among people in indoor environments. It plans in two layers that feed into each other, under the kinodynamic constraints of the robot:

- a **global planner** (RRT\*) finds a path through the map that avoids crowded areas, using a social heatmap built from the people seen over time;
- a **local kinodynamic planner** (SST) follows that path while keeping the people around comfortable. It gets feedback from the global path, from its own previous solution, and from a CNN that predicts where near-optimal positions and headings lie (learning-informed sampling).

It is composed of four packages:

- `kino_smf_move_base_mapping`: the world model, a 3D map (OctoMap) and a multilayer grid map with the obstacles, the people and the social heatmap.
- `kino_smf_move_base_planning`: the global and local planners.
- `kino_smf_move_base_control`: a path follower for differential-drive robots.
- `kino_smf_move_base_msgs`: the message and action definitions of the stack.

Each package has a launch file and an example configuration, see [Running](#running).

![FrameworkConnections](https://i.imgur.com/JgMGqJW.png)

- [Kino-SMF: Kinodynamic Social Multilayer Feedback Navigation](#kino-smf-kinodynamic-social-multilayer-feedback-navigation)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Building](#building)
    - [Running](#running)
  - [World Modeling (`kino_smf_move_base_mapping`)](#world-modeling-kino_smf_move_base_mapping)
    - [Parameters](#parameters)
    - [Subscribers](#subscribers)
    - [Publishers](#publishers)
    - [Services](#services)
  - [Online Social Robot Path Planning (`kino_smf_move_base_planning`)](#online-social-robot-path-planning-kino_smf_move_base_planning)
    - [Training and optimising the CNN](#training-and-optimising-the-cnn)
    - [Parameters](#parameters-1)
    - [Subscribers](#subscribers-1)
    - [Publishers](#publishers-1)
    - [Services](#services-1)
  - [Path Following Control (`kino_smf_move_base_control`)](#path-following-control-kino_smf_move_base_control)
    - [Parameters](#parameters-2)
    - [Subscribers](#subscribers-2)
    - [Publishers](#publishers-2)
  - [Citation](#citation)

## Installation

### Requirements

- Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html). The other ROS dependencies are installed with `rosdep` (see [Building](#building)).
- [pedsim_ros](https://github.com/CardiffUniversityComputationalRobotics/pedsim_ros) (`jazzy` branch), which provides `pedsim_msgs`.
- An NVIDIA GPU with the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) and [libtorch](https://pytorch.org/get-started/locally/), the C++ distribution of PyTorch, built for the same CUDA version. The planner runs its CNN with them.
- Optional: [TensorRT](https://developer.nvidia.com/tensorrt) and the [Torch-TensorRT](https://github.com/pytorch/TensorRT/releases) runtime, to load TensorRT-optimised models. Without them the planner only loads plain TorchScript models.
- Optional, only to [train or optimise the CNN](#training-and-optimising-the-cnn): Python 3 with PyTorch (`torch`, `torchvision`), `numpy` and `opencv-python`, plus Torch-TensorRT to optimise:

  ```bash
  python3 -m venv ~/kino_smf_venv && source ~/kino_smf_venv/bin/activate
  pip install torch torchvision numpy opencv-python
  pip install torch-tensorrt   # only to optimise
  ```

  On a Jetson, install the `torch`, `torchvision` and `torch-tensorrt` wheels built for your JetPack release instead.

The planner's `CMakeLists.txt` detects the architecture and looks for the GPU libraries where they are usually installed:

| | x86_64 (amd64) | Jetson (aarch64) |
|---|---|---|
| CUDA toolkit | `/usr/local/cuda`, or the newest `/usr/local/cuda-X.Y` | `/usr/local/cuda` (JetPack) |
| libtorch | `~/libtorch` (libtorch zip, cxx11 ABI, CUDA build), or `~/libtorch-*/libtorch` | `~/libtorch` as a symlink to the `site-packages/torch` of the pip wheel, or the torch `python3` imports |
| Torch-TensorRT | `~/libtorchtrt-*-x86_64-linux/torch_tensorrt` (release tarball) | `site-packages/torch_tensorrt`, next to torch |
| TensorRT | `/usr/lib/x86_64-linux-gnu` (apt) | `site-packages/tensorrt_libs` next to torch, or `/usr/lib/aarch64-linux-gnu` |

### Building

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone -b jazzy-devel https://github.com/CardiffUniversityComputationalRobotics/kino-smf-nav-stack.git
git clone -b jazzy https://github.com/CardiffUniversityComputationalRobotics/pedsim_ros.git
cd ~/ros2_ws
rosdep install --from-paths src/kino-smf-nav-stack src/pedsim_ros/pedsim_msgs --ignore-src -y
colcon build --packages-up-to kino_smf_move_base_planning kino_smf_move_base_control \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

The configure step of the planner prints the libraries it found (`kino_smf:` lines).

- Any location can be forced with `--cmake-args`: `-DCUDA_TOOLKIT_ROOT_DIR=`, `-DLIBTORCH_ROOT=`, `-DTORCH_TENSORRT_ROOT=`, `-DTENSORRT_ROOT=`. The environment variables `CUDA_HOME`, `LIBTORCH_ROOT`, `TORCH_TENSORRT_ROOT` and `TENSORRT_ROOT` work too.
- Torch-TensorRT is used when found: `-DKINO_SMF_TORCH_TENSORRT=AUTO` (default), `ON` (required) or `OFF`.
- When the prebuilt Torch-TensorRT needs a newer `libstdc++` than the system one, a newer one (e.g. the one shipped with Nsight Compute) is linked and installed next to the executable. `-DTORCH_TENSORRT_LIBSTDCXX=/path/to/libstdc++.so.6` forces one.

### Running

Each package has a launch file that starts its node with the example configuration of its `config` folder. Start them in this order, once the robot or the simulation is running: the mapper stops if it cannot get the transforms from `robot_frame` to the sensor frames within about 10 s.

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch kino_smf_move_base_mapping kino_smf_move_base_mapping.launch.py
ros2 launch kino_smf_move_base_planning kino_smf_move_base_planning.launch.py
ros2 launch kino_smf_move_base_control kino_smf_move_base_control.launch.py
```

Copy the configurations, adapt the topics, frames and velocity limits to your robot, and pass them with `mapping_config_file:=<file>`, `planning_config_file:=<file>` and `control_config_file:=<file>`. In simulation, set `use_sim_time: true` in them. The example configurations connect the three nodes, and take the navigation goals (`geometry_msgs/msg/PoseStamped`) on `/goal_pose`, where the 2D Goal Pose tool of RViz publishes them.

The example configuration of the planner loads the trained CNN from `~/kino_smf_models/model_xy_yaw_trt_fp16.ts`. Download it from [Zenodo](https://zenodo.org/records/24367228), see [Training and optimising the CNN](#training-and-optimising-the-cnn).

## World Modeling (`kino_smf_move_base_mapping`)

This package is the perception of the robot. It builds a 3D map of the space with OctoMap from a laser scan and a point cloud (e.g. from a depth camera), and combines it in a multilayer grid map with the people around the robot: the obstacles, the comfort space of the people and a social heatmap of the crowded areas.

### Parameters

- map_frame (string, default: "map")

  Frame of the maps.

- fixed_frame (string, default: "fixed_frame")

  Fixed frame in which the sensor data is integrated, e.g. `odom`.

- robot_frame (string, default: "robot_frame")

  Base frame of the robot.

- resolution (double, default: 1.0)

  Resolution of the OctoMap and of the grid map [m].

- offline_octomap_path (string, default: "")

  If set, this OctoMap (`.bt`) is used and no mapping is done.

- mapping_max_range (double, default: 5.0)

  Maximum range of the sensor data used to build the map [m].

- min_z_pc (double, default: 0.05)

  Obstacles are mapped above this height [m], and above `min_z_pc` + `pc_min_z_offset` for the point cloud.

- max_z_pc (double, default: 1.0)

  The point cloud maps obstacles below this height [m].

- visualize_free_space (bool, default: false)

  Publish the maps on `octomap_map` and `social_grid_map`.

- rviz_timer (double, default: 0.0)

  Period of the visualization topics [s].

- odometry_topic (string, default: "/odometry_topic")

  Odometry of the robot.

- erase_map_topic (string, default: "erase_map")

  A `true` on this topic clears the map.

- laser_scan_topic (string, default: "")

  Laser scan used to build the map. Both sensors are needed when mapping online.

- laser_scan_frame (string, default: "")

  Frame of the laser scan. The node waits at start-up for its transform to `robot_frame`.

- minimum_range (double, default: -1.0)

  Laser returns closer than this are ignored [m]. -1.0 disables it.

- apply_filter (bool, default: false)

  Remove isolated single returns from the laser scan.

- add_max_ranges (bool, default: false)

  Use the laser readings without a return to clear free space.

- laser_zero_is_no_return (bool, default: false)

  Count 0.0 laser ranges as readings without a return.

- point_cloud_topic (string, default: "")

  Point cloud (e.g. from a depth camera) used to build the map.

- point_cloud_frame (string, default: "")

  Frame of the point cloud. The node waits at start-up for its transform to `robot_frame`.

- pc_min_z_offset (double, default: 0.2)

  The point cloud only maps above `min_z_pc` + this offset [m].

- pc_max_obstacle_range (double, default: 3.5)

  Points farther than this only clear free space [m].

- pc_voxel_min_points (int, default: 3)

  Points a voxel needs in one point cloud to be marked as occupied, which filters out noise.

- pc_range_noise (double, default: 0.004)

  Depth noise of the point cloud, sigma = `pc_range_noise` x range². Free space is only cleared up to 3 sigma before each point.

- prob_hit (double, default: 0.7), prob_miss (double, default: 0.4)

  Occupancy probability of a voxel that a return hits, and of a voxel that a ray goes through.

- clamping_thres_min (double, default: 0.1192), clamping_thres_max (double, default: 0.971)

  Bounds of the occupancy probability of a voxel.

- occupancy_thres (double, default: 0.75)

  Occupancy probability above which a voxel is an obstacle.

- obstacle_decay_time (double, default: 0.0)

  Occupied voxels that are not observed for this long are cleared [s]. 0.0 disables it.

- social_agents_topic (string, default: "/pedsim_simulator/simulated_agents")

  States of the people around the robot.

- social_agent_radius (double, default: 0.4)

  Radius of a person [m].

- agent_filter_radius (double, default: 0.6)

  Sensor returns this close to a person only clear free space, so that people are not mapped as obstacles [m].

- social_relevance_validity_checking (bool, default: false)

  Only consider the people in the field of view of the robot (the relevant agents).

- robot_distance_view_max (double, default: 6.0), robot_distance_view_min (double, default: 1.5)

  Range of the field of view [m]. It grows with the speed of the robot, from the minimum when stopped to the maximum at `robot_velocity_threshold`.

- robot_angle_view (double, default: 1.57)

  A person is in the field of view when within this angle of the heading of the robot, on either side [rad].

- robot_velocity_threshold (double, default: 0.3)

  Speed of the robot at which the field of view reaches `robot_distance_view_max` [m/s].

- social_heatmap_decay_factor (double, default: 65.0)

  How fast the social heatmap forgets: a person seen t seconds ago weighs 100·exp(−factor·t/60).

- social_comfort_amplitude (double, default: 6.0)

  Amplitude of the personal space cost around each person.

### Subscribers

A topic written as a parameter name is set by that parameter.

- `social_agents_topic` (`pedsim_msgs/msg/AgentStates`)

  Position, orientation and velocity of the people.

- `odometry_topic` (`nav_msgs/msg/Odometry`)

  Odometry of the robot.

- `laser_scan_topic` (`sensor_msgs/msg/LaserScan`) and `point_cloud_topic` (`sensor_msgs/msg/PointCloud2`)

  Sensor data used to build the map.

- `erase_map_topic` (`std_msgs/msg/Bool`)

  Clears the map on `true`.

### Publishers

- `octomap_map` (`visualization_msgs/msg/MarkerArray`)

  The 3D map, for visualization.

- `social_grid_map` (`grid_map_msgs/msg/GridMap`)

  The grid map, for visualization.

- `relevant_agents` (`pedsim_msgs/msg/AgentStates`)

  The people in the field of view of the robot, used by the planner.

### Services

- `get_grid_map` (`grid_map_msgs/srv/GetGridMap`)

  Returns the grid map, used by the planner.

- `get_binary` (`octomap_msgs/srv/GetOctomap`)

  Returns the 3D map.

- `save_binary` and `save_full` (`std_srvs/srv/Empty`)

  Save the 3D map in the home directory, as `map_laser_octomap.bt` (binary) or `map_laser_octomap.ot` (full probabilities).

## Online Social Robot Path Planning (`kino_smf_move_base_planning`)

This package finds socially acceptable paths for the robot. In every planning cycle, the global planner plans to the goal over the whole map, and the local kinodynamic planner plans the motions of the robot towards a local goal on the global path, guided by the CNN.

### Training and optimising the CNN

The planner guides its sampling with one CNN (`XYYawCNN`, a ResNet-50 trunk with two heads). On a 10 m x 10 m grid around the robot (100 x 100 cells of 0.1 m), it predicts where near-optimal paths pass (`xy`) and with which heading (`yaw`). The planner loads it from the `xy_yaw_model_path` parameter (default `~/kino_smf_models/model_xy_yaw_trt_fp16.ts`, a leading `~` is expanded). The scripts below are in [`kino_smf_move_base_planning/scripts`](kino_smf_move_base_planning/scripts); the other scripts there are evaluations and earlier experiments.

**Dataset and trained model.** The dataset and the trained model of the paper are on [Zenodo](https://zenodo.org/records/24367228). `ral_dataset.zip` is the dataset, in the layout below. `xy_yaw_best_final.pth` is the checkpoint of the trained model, and `model_xy_yaw_trt_fp16.ts`, `model_xy_yaw_trt_tf32.ts` and `model_xy_yaw_trt.ts` are TensorRT models optimised from it. Put the models in `~/kino_smf_models`, where the planner and `optimize_xy_yaw_jetson.py` look for them by default. A TensorRT model only runs on the GPU model and TensorRT version it was built with, so if these do not load on your machine, optimise the checkpoint there (see **Optimise** below).

**Dataset.** Each sample is three 100 x 100 grayscale PNG files named `<YYYY-MM-DD>_<HH-MM-SS>_<start yaw>_<goal yaw>.png`, with the yaws in radians in [0, 2π). A pixel stores value / scale x 255:

| File | Content |
|---|---|
| `test/<name>.png` | input grid, scale 6: 0 free, 1 obstacle, 2 start (centre), 3 goal, 5 social agent |
| `train/path/<name>.png` | the same grid with the cells of the planned path set to 4 |
| `train/orientation/<name>yaw.png` | heading of the path in its cells, scale 8, and 7 (no data) elsewhere |

The [`dataset`](https://github.com/CardiffUniversityComputationalRobotics/kino-smf-nav-stack/tree/dataset) branch of this repository records these samples while the planner runs. Set the output directory in its `planning_framework_main.cpp` first.

**Train.** A GPU is strongly recommended. The first run downloads the ImageNet weights of ResNet-50 and caches the preprocessed dataset in `dataset_cache.npz`:

```bash
SCRIPTS=~/ros2_ws/src/kino-smf-nav-stack/kino_smf_move_base_planning/scripts
mkdir -p ~/kino_smf_training && cd ~/kino_smf_training
python3 $SCRIPTS/training_script_xy_yaw.py --dataset /path/to/dataset
```

The samples are split 80/10/10 into train, validation and test with a fixed seed. Training uses AdamW with warmup and cosine decay, and stops early when the validation recall has not improved for `--patience` epochs (default 25). The best checkpoint is kept as `trained/xy_yaw_best.pth`, the metrics on the held-out test split are printed, and the model is exported as `trained/model_xy_yaw.pt`, a TorchScript model the planner loads as is. `--epochs`, `--batch` and `--lr` change the schedule, `--quick` runs a 3-epoch smoke test, and `--cache` / `--out-dir` move the outputs.

**Optimise** the model with TensorRT for faster inference. Do it on the machine that runs the planner: a TensorRT engine only runs on the GPU model and TensorRT version it was built with.

```bash
cd ~/kino_smf_training
# Torch-TensorRT before 2.14 (TensorRT 10)
python3 $SCRIPTS/optimize_xy_yaw.py --ckpt trained/xy_yaw_best.pth --out trained/model_xy_yaw_trt_fp16.ts
# Torch-TensorRT 2.14 or later (TensorRT 11), e.g. on a Jetson
python3 $SCRIPTS/optimize_xy_yaw_jetson.py --ckpt trained/xy_yaw_best.pth --out trained/model_xy_yaw_trt_fp16.ts --dataset-cache dataset_cache.npz
```

Both build a static batch-1 FP16 engine (`--precision tf32` or `fp32_strict` for the others) and save it as TorchScript. They then compare it with the PyTorch model on held-out samples of `dataset_cache.npz`, and print the latency of both. Without `--dataset-cache`, `optimize_xy_yaw_jetson.py` compares on synthetic grids, for a machine that does not have the dataset. Set `xy_yaw_model_path` to the `.ts` file; the planner must be built with Torch-TensorRT to load it.

### Parameters

- world_frame (string, default: "map")

  Frame in which the planning is done.

- planner_name (string, default: "RRTstarMod")

  Global planner: `RRT`, `RRTstar`, `RRTstarMod` or `PRMstar`.

- optimization_objective (string, default: "SocialHeatmap")

  Objective of the global planner: `PathLength` or `SocialHeatmap`.

- planning_bounds_x (double list, default: [-50.0, 50.0]), planning_bounds_y (double list, default: [-50.0, 50.0])

  Planning bounds, [min, max] in each axis [m].

- dynamic_bounds (bool, default: true)

  Fit the planning bounds to the start and goal of each navigation query.

- timer_period (double, default: 0.52)

  Period of the planning cycle [s].

- solving_time (double, default: 0.5)

  Planning time of a cycle, shared by the global and local planners [s].

- global_time_percent (double, default: 10.0)

  Share of `solving_time` given to the global planner [%].

- reuse_last_best_solution (bool, default: true)

  The global planner starts from its last best solution.

- opport_collision_check (bool, default: true)

  Use opportunistic collision checking.

- motion_cost_interpolation (bool, default: false)

  Interpolate the cost along each motion.

- xy_goal_tolerance (double, default: 0.1), yaw_goal_tolerance (double, default: 0.75)

  Position [m] and heading [rad] tolerance of the goal.

- robot_base_radius (double, default: 0.35)

  Radius of the base of the robot, for collision checking [m].

- visualize_tree (bool, default: false)

  Publish the tree of the planner for visualization.

- local_planner_name (string, default: "SST")

  Local kinodynamic planner: `SST` or `RRT`.

- local_optimization_objective (string, default: "SocialComfort")

  Objective of the local planner: `PathLength`, `SocialHeatmap` or `SocialComfort`.

- local_path_range (double, default: 4.0)

  Distance along the global path from the robot to the local goal [m].

- local_xy_goal_tolerance (double, default: 1.0)

  Position tolerance of the local goal [m].

- local_reuse_last_best_solution (bool, default: true)

  The local planner starts from its last best solution.

- local_use_social_heatmap (bool, default: false)

  The local planner also takes the social heatmap into account.

- max_trans_vel (double, default: 0.3), max_rot_vel (double, default: 1.2)

  Linear [m/s] and angular [rad/s] velocity limits of the local planner.

- odometry_topic (string, default: "/odom")

  Odometry of the robot.

- query_goal_topic (string, default: "/kino_smf_move_base_planner/query_goal")

  Navigation goals.

- solution_path_topic (string, default: "/kino_smf_move_base_planner/solution_path")

  Path sent to the controller.

- grid_map_service (string, default: "/get_grid_map")

  Service of the mapper that returns the grid map.

- xy_yaw_model_path (string, default: "~/kino_smf_models/model_xy_yaw_trt_fp16.ts")

  CNN model, see [Training and optimising the CNN](#training-and-optimising-the-cnn). A leading `~` is expanded.

- use_position_cnn (bool, default: true)

  Guide the sampling of the local planner with the positions predicted by the CNN. The CNN is only loaded when this is true.

- use_orientation_cnn (bool, default: true)

  Also take the headings of the samples from the CNN.

- rejection_chance (double, default: 0.5)

  Probability of checking a sample against the positions predicted by the CNN, and rejecting it when it is outside of them.

- orientation_chance (double, default: 1.0)

  Probability of taking the heading of a sample from the CNN.

### Subscribers

A topic written as a parameter name is set by that parameter.

- `query_goal_topic` (`geometry_msgs/msg/PoseStamped`)

  Navigation goals.

- `odometry_topic` (`nav_msgs/msg/Odometry`)

  Odometry of the robot.

- `relevant_agents` (`pedsim_msgs/msg/AgentStates`)

  The people in the field of view of the robot, from the mapper.

- `stop_motion` (`std_msgs/msg/Bool`)

  Cancels the current goal on `true`.

### Publishers

- `solution_path_topic` (`nav_msgs/msg/Path`)

  Path for the controller to follow.

- `solution_path` and `local_solution_path` (`visualization_msgs/msg/Marker`)

  Global and local paths, for visualization.

- `query_goal_pose_rviz` (`geometry_msgs/msg/PoseStamped`) and `query_goal_radius_rviz` (`visualization_msgs/msg/Marker`)

  Current goal and its tolerance, for visualization.

- `global_planner_bounds` (`visualization_msgs/msg/Marker`)

  Bounds of the global planner, for visualization.

- `kino_smf_num_nodes` (`std_msgs/msg/Int32`)

  Number of nodes of the local planner.

- `goal_reached` (`std_msgs/msg/Bool`)

  `true` when the goal is reached.

- `sweep_motion` (`std_msgs/msg/Bool`)

  Asks the controller to turn the robot in place when a navigation query starts, to see around it.

### Services

- Calls `grid_map_service` (`grid_map_msgs/srv/GetGridMap`) to get the grid map of the mapper.

## Path Following Control (`kino_smf_move_base_control`)

This package sends the velocity commands that make a differential-drive robot follow the path of the planner.

### Parameters

- max_trans_vel (double, default: 0.25), max_rot_vel (double, default: 1.5)

  Linear [m/s] and angular [rad/s] velocity limits.

- goal_tolerance (double, default: 0.1)

  Distance to the goal at which it is reached [m].

- robot_frame (string, default: "base_footprint")

  Base frame of the robot.

- odometry_topic (string, default: "/odom")

  Odometry of the robot.

- query_goal_topic (string, default: "/goal_pose")

  Navigation goals.

- control_path_topic (string, default: "/path")

  Path to follow, the `solution_path_topic` of the planner.

- control_output_topic (string, default: "/cmd_vel")

  Velocity commands.

- stop_motion_topic (string, default: "/stop_motion")

  Stops the robot on `true`.

- sweep_motion_topic (string, default: "/sweep_motion")

  Turns the robot in place, one full turn, on request of the planner.

### Subscribers

A topic written as a parameter name is set by that parameter.

- `control_path_topic` (`nav_msgs/msg/Path`)

  Path to follow.

- `query_goal_topic` (`geometry_msgs/msg/PoseStamped`)

  Navigation goals.

- `odometry_topic` (`nav_msgs/msg/Odometry`)

  Odometry of the robot.

- `stop_motion_topic` and `sweep_motion_topic` (`std_msgs/msg/Bool`)

  Stop and in-place turn requests.

- `/scan` (`sensor_msgs/msg/LaserScan`)

  Laser scan used to detect the obstacles close to the robot.

### Publishers

- `control_output_topic` (`geometry_msgs/msg/Twist`)

  Velocity commands.

- `/robot_is_moving` (`std_msgs/msg/Bool`)

  Whether the robot is moving.

- `/goal_reached` (`std_msgs/msg/Bool`)

  `true` when the goal is reached.

## Citation

This work is part of the paper [Social Robot Navigation Under Kinodynamic Constraints Using Learning-Informed Sampling for Indoor Environments](https://ieeexplore.ieee.org/document/11543254), published in IEEE Robotics and Automation Letters (2026). If you use it, please cite:

```bibtex
@ARTICLE{Silva2026,
  author={Silva, Steven and Romero-Cano, Victor and Hernández, Juan David},
  journal={IEEE Robotics and Automation Letters},
  title={Social Robot Navigation Under Kinodynamic Constraints Using Learning-Informed Sampling for Indoor Environments},
  year={2026},
  volume={11},
  number={7},
  pages={9000-9007},
  keywords={Robots;Modeling;Planning;Convolutional neural networks;Navigation;Path planning;Learning (artificial intelligence);Timing;Social robots;Algorithms;Human-aware motion planning;motion and path planning},
  doi={10.1109/LRA.2026.3699250}}
```
