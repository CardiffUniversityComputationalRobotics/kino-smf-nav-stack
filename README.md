# Fedback Social Comfort Robot Navigation Framework

This is an online social robot navigation framework for indoor social scenarios that uses a Social Heatmap to represent crowded areas in an environment and also uses a multilayer technique for the planning module.

It is composed of four different packages:

- `kino_smf_move_base_control`: it is a simple differential control method.
- `kino_smf_move_base_mapping`: is in charge of the perception of the space.
- `kino_smf_move_base_planning`: responsible for finding solution paths for the navigation query requested.
- `kino_smf_move_base_msgs`: contains the messages needed for the framework and the start-goal queries.

To run the framework, there is a launch file example in every package with example configurations.

![FrameworkConnections](https://i.imgur.com/JgMGqJW.png)

- [Fedback Social Comfort Robot Navigation Framework](#fedback-social-comfort-robot-navigation-framework)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Building](#building)
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
    - [Actions](#actions)
  - [Path Following Control (`kino_smf_move_base_control`)](#path-following-control-kino_smf_move_base_control)
    - [Parameters](#parameters-2)
    - [Subscribers](#subscribers-2)
    - [Publishers](#publishers-2)

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

## World Modeling (`kino_smf_move_base_mapping`)

This package is in charge of the perception for the robot. For that, a depth camera is used, which helps to generate a 3D map of the space by using Octomap.

This 3D map is combined in a multilayer GridMap with the position of the social agents detected in the space.

Alditionally, this package also provides an idea of the crowded areas in the social environment.

### Parameters

- resolution (double, default: 1.0)

  Octomap and Grid Map resolution.

- map_frame (string, default: "map")

- fixed_frame (string, default: "fixed_frame")

  Fixed frame considered for the mapping, can be `odom` for example.

- robot_frame (string, default: "/robot_frame")

  Base frame considered from the robot for the mapping.

- oflline_octomap_path (string, default: "")

  If defined, then the given map is used and no ther mapping is done.

- visualize_free_space (bool, default: True)

  Wether you would like to see the 3D map and grid_map. If `False`, nothing is published at `/kino_smf_move_base_mapping/social_grid_map` and `/kino_smf_move_base_mapping/octomap_map`.

- mapping_max_range (double, default: 5.0)

  Maximum distance to generate the 3D map from the obtained depth points.

- odometry_topic (string, default: "/odometry_topic")

- rviz_timer (double, default: 0.0)

  The time delay to publish messages to RViz. Not used if `visualize_free_space` is `False`.

- point_cloud_topics (list: string, default: empty)

  List of pointcloud topics used to generate the 3D map.

- point_cloud_frames (list: string, default: empty)

  List of frames from pointcloud topics in order respectively.

- social_agents_topic (string, default: "/pedsim_simulator/simulated_agents")

  Topic with social agents states.

- social_agent_radius (double, default: 0.4)

  Radius considered for the social agents.

- social_relevance_validity_checking (bool, default: False)

  Whether or not it is desired to only consider the relevant agents.

- robot_distance_view_max (double, default: 6.0)

  Maximum distance for the robot to consider social agents.

- robot_distance_view_min (double, default: 1.5)

  Mininum distance for the robot to consider social agents.

- robot_angle_view (double, default: 1.57)

  Maximum angle in which the social agents are considered from the field of view of the robot. Defined in radians.

- robot_velocity_threshold (double, default: 0.3)

  Maximum velocity that the robot can have.

- social_heatmap_decay_factor (double, default: 65)

  It is recommended a value in between 60 and 75.

### Subscribers

The name of the subscribers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /pedsim_simulator/simulated_agents ([pedsim_msgs/AgentStates](https://github.com/CardiffUniversityComputationalRobotics/pedsim_ros/blob/noetic-devel/pedsim_msgs/msg/AgentStates.msg))

  Position, orientation, velocity and other states of the social agents.

- /pepper/camera/depth/points ([sensor_msgs/PointCloud2](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html))

  Depth points from depth camera.

- /pepper/odom_groundtruth ([nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html))

  Robot odometry (in this case it was Pepper robot).

### Publishers

The name of the publishers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /kino_smf_move_base_mapping/octomap_map ([visualization_msgs/MarkerArray](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/MarkerArray.html))

  Visual representation of the 3D map created.

- /kino_smf_move_base_mapping/social_grid_map ([grid_map_msgs/GridMap](http://docs.ros.org/en/kinetic/api/grid_map_msgs/html/msg/GridMap.html))

  Grid map with 4 layers, one for obstacles and one including obstacles and social agents, one for comfort and another for social heatmap.

### Services

- /kino_smf_move_base_mapping/get_grid_map ([grid_map_msgs/GetGridMap](http://docs.ros.org/en/indigo/api/grid_map_msgs/html/srv/GetGridMap.html))

  Fetches the grid map from the World Modeling.

## Online Social Robot Path Planning (`kino_smf_move_base_planning`)

This package is in charge of finding socially acceptable solution paths for the robot to follow.

### Training and optimising the CNN

The planner guides its sampling with one CNN (`XYYawCNN`, a ResNet-50 trunk with two heads). On a 10 m x 10 m grid around the robot (100 x 100 cells of 0.1 m), it predicts where near-optimal paths pass (`xy`) and with which heading (`yaw`). The planner loads it from the `xy_yaw_model_path` parameter (default `~/tidup_models/model_xy_yaw_trt_fp16.ts`, a leading `~` is expanded). The scripts below are in [`kino_smf_move_base_planning/scripts`](kino_smf_move_base_planning/scripts); the other scripts there are evaluations and earlier experiments.

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

All of the following parameters have to be defined since they have no default values.

- world_frame (string)

  Main parent frame to be used as reference.

- planner_name (string)

  Planner to be used for global, can be RRT, RRTstar or PRMstar.

- local_planner_name (string)

  Planner to be used for local, can be RRT, RRTstar or PRMstar, InformedRRTstarMod, InformedRRTstar.

- planning_bounds_x (list: [x_min, x_max])

  Limits for planning in the X axis.

- planning_bounds_y (list: [y_min, y_max])

  Limits for planning in the Y axis.

- dynamic_bounds (bool)

  Whether it is wanted to change the planning bounds dinamically depending on the navigation query.

- start_state (list, [X, Y, Yaw])

  Define start state for navigation query if the action is not going to be used.

- goal_state (list, [X, Y, Yaw])

  Define goal state for navigation query if no action is used.

- timer_period (double)

  Controls the time for sleeps in the node.

- solving_time (double)

  Time given to the planner to solve the navigation query, both for the global and the local layer.

- opport_collision_check (bool)

  Whether it is wanted to use opportunistic collision checking or not.

- reuse_last_best_solution (bool)

  Whether it is wanted to apply the reuse of the last best known solution for the global planner.

- local_reuse_last_best_solution (bool)

  Whether it is wanted to apply the reuse of the last best known solution for the local planner.

- optimization_objective (string)

  Defines which optimization objective to use is desired for the global planner. Currently, only supported PathLength and SocialHeatmap.

- local_optimization_objective (string)

  Defines which optimization objective to use is desired for the local planner. Currently, only supported PathLength and SocialComfort.

- motion_cost_interpolation (bool)

  If motion cost interpolation is desired.

- xy_goal_tolerance (double)

  Tolerance in the XY axis for the global goal state.

- local_xy_goal_tolerance (double)

  Tolerance in the XY axis for the local goal state.

- visualize_tree: (bool)

  Whether it is desired to visualize or not the tree generated by the planner.

- grid_map_service (string)

  Service name to get the GridMap.

- control_active_topic (string)

  Topic to communicate with the control module to know when control is ready.

- robot_base_radius (double)

  Radius of the base of the robot to be considered for the planning.

- odometry_topic (string)

- query_goal_topic (string)

  Topic to request a query without using actions.

- goto_action (string)

  Action name to run a navigation query.

- local_path_range (double)

  Distance from the robot position to the decided local goal.

- global_time_percent (double)

  Percentage given to the global planner from the time budget.

### Subscribers

The name of the subscribers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /kino_smf_move_base_planner/query_goal ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html))
- /pepper/odom_groundtruth ([nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html))

  Robot odometry.

- /control_active_topic ([std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html))

### Publishers

The name of the publishers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /kino_smf_move_base_planner/kino_smf_move_base_solution_path ([kino_smf_move_base_msgs/Path2D](https://github.com/CardiffUniversityComputationalRobotics/smf-nav-stack/blob/world_modeling/kino_smf_move_base_msgs/msg/Path2D.msg))

  Solution path found by the planner. It is passed to the control module.

- /kino_smf_move_base_planner/smf_num_nodes ([std_msgs/Int32](http://docs.ros.org/en/melodic/api/std_msgs/html/msg/Int32.html))

  Number of valid nodes sampled by the local planner.

- /kino_smf_move_base_planner/query_goal_pose_rviz ([geometry_msgs/PoseStamped](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PoseStamped.html))

  Shows in RViz the query pose requested to the planner.

- /kino_smf_move_base_planner/query_goal_radius_rviz ([visualization_msgs/Marker](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html))

  Shows in RViz the radius of the query requested to the planner.

- /kino_smf_move_base_planner/solution_path ([visualization_msgs/Marker](http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html))

  Visual solution path to be seen in RViz.

### Actions

- `/goto_action` ([kino_smf_move_base_msgs/GoTo2D](https://github.com/CardiffUniversityComputationalRobotics/smf-nav-stack/blob/world_modeling/kino_smf_move_base_msgs/action/Goto2D.action))

  Action to request navigation query. The name of the action server depends on the parameter `goto_action`.

## Path Following Control (`kino_smf_move_base_control`)

This module is in charge of sending the velocities to the robot according to the path obtained so that the robot goes through that path.

### Parameters

- max_vel (double, default: 0.1)

  Maximum velocity the robot can have in the X axis.

- min_vel (double, default: 0.05)

  Minimum velocity for the robot to move in the X axis.

- max_turn_rate (double, default: 0.5)

  Maximum velocity with which the robot can rotate.

- min_turn_rate (double, default: 0.5)

  Minimum velocity for the robot to rotate.

- controller_hz (double, default: 100)

  Frequency in which velocities are sent

- control_path_topic (string, default: "/control_path_topic")

  Topic in which the desired path to follow is published.

- control_output_topic (double, default: "/control_output_topic")

  Topic in which the velocities are published.

- odometry_topic (string, default: "/odometry_topic")

### Subscribers

The name of the subscribers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /kino_smf_move_base_planner/kino_smf_move_base_solution_path ([kino_smf_move_base_msgs/Path2D](https://github.com/CardiffUniversityComputationalRobotics/smf-nav-stack/blob/world_modeling/kino_smf_move_base_msgs/msg/Path2D.msg))
- /pepper/odom_groundtruth ([nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html))

### Publishers

The name of the publishers' topics are just defined as an example, but they may be configured using the parameters defined before.

- /control_active_topic ([std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html))
- /pepper/cmd_vel ([geometry_msgs/Twist](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Twist.html))
