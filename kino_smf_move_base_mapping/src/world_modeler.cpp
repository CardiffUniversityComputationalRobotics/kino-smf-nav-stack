/*! \file octomap_laser_scan.cpp
 * \brief Merge data from different laser_scan messages to incrementally build
 * an WorldModeler.
 *
 * \date November 07, 2022
 * \author Juan David Hernandez Vega, HernandezVegaJ@cardiff.ac.uk
 * \author Steven Alexander Silva Mendoza, silvas1@cardiff.ac.uk
 *
 * \details Purpose: Merge data from different laser_scan messages to
 * incrementally build an Octomap.
 *  Based on laser_octomap (Guillem Vallicrosa & J.D. Hernandez Vega, University
 * of Girona)
 */

#include <world_modeler.hpp>

//!  Constructor.
/*!
 * Load map parameters.
 * Subscribers to odometry and laser scan
 * Publishers to visualize the WorldModeler.
 */
WorldModeler::WorldModeler()
    : Node("world_modeler_node"),
      fixed_frame_("fixed_frame"),
      robot_frame_("robot_frame"),
      map_frame_("map"),
      odometry_topic_("/odometry_topic"),
      offline_octomap_path_(""),
      octree_(NULL),
      octree_resol_(1.0),
      mapping_max_range_(5.0),
      initialized_(false),
      visualize_free_space_(false),
      rviz_timer_(0.0),
      robot_distance_view_max_(6.0),
      robot_distance_view_min_(1.5),
      robot_angle_view_(1.57),
      robot_velocity_threshold_(0.3),
      social_agent_radius_(0.4),
      social_agents_topic_("/pedsim_simulator/simulated_agents"),
      social_relevance_validity_checking_(false),
      social_heatmap_decay_factor_(65.0),
      min_z_pc_(0.05),
      max_z_pc_(1.0),
      social_comfort_amplitude_(6.0),
      orientation_drift_(0.0),
      position_drift_(0.0),
      apply_filter_(false),
      add_max_ranges_(false),
      add_rays_(false),
      skip_sensor_callbacks_after_erase_(false),
      minimum_range_(-1.0)
{
    //=======================================================================
    // Get parameters
    //=======================================================================
    this->declare_parameter("add_rays", add_rays_);
    this->declare_parameter("resolution", octree_resol_);
    this->declare_parameter("map_frame", map_frame_);
    this->declare_parameter("fixed_frame", fixed_frame_);
    this->declare_parameter("robot_frame", robot_frame_);
    this->declare_parameter("offline_octomap_path", offline_octomap_path_);
    this->declare_parameter("visualize_free_space", visualize_free_space_);
    this->declare_parameter("odometry_topic", odometry_topic_);
    this->declare_parameter("rviz_timer", rviz_timer_);
    this->declare_parameter("laser_scan_topic", laser_scan_topic_);
    this->declare_parameter("laser_scan_frame", laser_scan_frame_);
    this->declare_parameter("point_cloud_topic", point_cloud_topic_);
    this->declare_parameter("point_cloud_frame", point_cloud_frame_);
    this->declare_parameter("mapping_max_range", mapping_max_range_);
    this->declare_parameter("robot_distance_view_max", robot_distance_view_max_);
    this->declare_parameter("robot_distance_view_min", robot_distance_view_min_);
    this->declare_parameter("robot_angle_view", robot_angle_view_);
    this->declare_parameter("robot_velocity_threshold", robot_velocity_threshold_);
    this->declare_parameter("erase_map_topic", rclcpp::ParameterValue(std::string("erase_map")));
    this->declare_parameter("social_agent_radius", social_agent_radius_);
    this->declare_parameter("social_agents_topic", social_agents_topic_);
    this->declare_parameter("social_relevance_validity_checking", social_relevance_validity_checking_);
    this->declare_parameter("social_heatmap_decay_factor", social_heatmap_decay_factor_);
    this->declare_parameter("min_z_pc", min_z_pc_);
    this->declare_parameter("max_z_pc", max_z_pc_);
    this->declare_parameter("social_comfort_amplitude", social_comfort_amplitude_);
    this->declare_parameter("apply_filter", apply_filter_);
    this->declare_parameter("add_max_ranges", add_max_ranges_);
    this->declare_parameter("laser_zero_is_no_return", laser_zero_is_no_return_);
    this->declare_parameter("minimum_range", minimum_range_);
    this->declare_parameter("prob_hit", prob_hit_);
    this->declare_parameter("prob_miss", prob_miss_);
    this->declare_parameter("clamping_thres_min", clamping_thres_min_);
    this->declare_parameter("clamping_thres_max", clamping_thres_max_);
    this->declare_parameter("occupancy_thres", occupancy_thres_);
    this->declare_parameter("pc_min_z_offset", pc_min_z_offset_);
    this->declare_parameter("pc_max_obstacle_range", pc_max_obstacle_range_);
    this->declare_parameter("pc_voxel_min_points", pc_voxel_min_points_);
    this->declare_parameter("pc_range_noise", pc_range_noise_);
    this->declare_parameter("agent_filter_radius", agent_filter_radius_);
    this->declare_parameter("obstacle_decay_time", obstacle_decay_time_);

    this->get_parameter("add_rays", add_rays_);
    this->get_parameter("resolution", octree_resol_);
    this->get_parameter("map_frame", map_frame_);
    this->get_parameter("fixed_frame", fixed_frame_);
    this->get_parameter("robot_frame", robot_frame_);
    this->get_parameter("offline_octomap_path", offline_octomap_path_);
    this->get_parameter("visualize_free_space", visualize_free_space_);
    this->get_parameter("odometry_topic", odometry_topic_);
    this->get_parameter("rviz_timer", rviz_timer_);
    this->get_parameter("laser_scan_topic", laser_scan_topic_);
    this->get_parameter("laser_scan_frame", laser_scan_frame_);
    this->get_parameter("point_cloud_topic", point_cloud_topic_);
    this->get_parameter("point_cloud_frame", point_cloud_frame_);
    this->get_parameter("mapping_max_range", mapping_max_range_);
    this->get_parameter("robot_distance_view_max", robot_distance_view_max_);
    this->get_parameter("robot_distance_view_min", robot_distance_view_min_);
    this->get_parameter("robot_angle_view", robot_angle_view_);
    this->get_parameter("robot_velocity_threshold", robot_velocity_threshold_);
    this->get_parameter("erase_map_topic", erase_map_topic_);
    this->get_parameter("social_agent_radius", social_agent_radius_);
    this->get_parameter("social_agents_topic", social_agents_topic_);
    this->get_parameter("social_relevance_validity_checking", social_relevance_validity_checking_);
    this->get_parameter("social_heatmap_decay_factor", social_heatmap_decay_factor_);
    this->get_parameter("min_z_pc", min_z_pc_);
    this->get_parameter("max_z_pc", max_z_pc_);
    this->get_parameter("social_comfort_amplitude", social_comfort_amplitude_);
    this->get_parameter("apply_filter", apply_filter_);
    this->get_parameter("add_max_ranges", add_max_ranges_);
    this->get_parameter("laser_zero_is_no_return", laser_zero_is_no_return_);
    this->get_parameter("minimum_range", minimum_range_);
    this->get_parameter("prob_hit", prob_hit_);
    this->get_parameter("prob_miss", prob_miss_);
    this->get_parameter("clamping_thres_min", clamping_thres_min_);
    this->get_parameter("clamping_thres_max", clamping_thres_max_);
    this->get_parameter("occupancy_thres", occupancy_thres_);
    this->get_parameter("pc_min_z_offset", pc_min_z_offset_);
    this->get_parameter("pc_max_obstacle_range", pc_max_obstacle_range_);
    this->get_parameter("pc_voxel_min_points", pc_voxel_min_points_);
    this->get_parameter("pc_range_noise", pc_range_noise_);
    this->get_parameter("agent_filter_radius", agent_filter_radius_);
    this->get_parameter("obstacle_decay_time", obstacle_decay_time_);

    // frame of the agent poses until an agent states message says otherwise
    agent_states_frame_ = fixed_frame_;

    // use_sim_time comes from the launch/config: forcing it on with no /clock (real
    // robot) leaves this node's clock at 0, so its stamps and timeouts break

    //=======================================================================
    // Transforms TF and catch the static transform from vehicle to laser_scan
    // sensor
    //=======================================================================

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_buffer_->setUsingDedicatedThread(true);
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    auto create_timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(),
        this->get_node_timers_interface());

    tf_buffer_->setCreateTimerInterface(create_timer_interface);

    // =============================
    // ! WAITING FOR LASER TRANSFORM
    // =============================
    int count(0);
    geometry_msgs::msg::TransformStamped transform;
    initialized_ = false;
    do
    {
        try
        {
            transform = tf_buffer_->lookupTransform(robot_frame_, laser_scan_frame_, tf2::TimePointZero, tf2::durationFromSec(1.0));
            initialized_ = true;
        }
        catch (tf2::TransformException &ex)
        {

            count++;
            RCLCPP_WARN(this->get_logger(), "Cannot find tf from %s to %s: %s",
                        robot_frame_.c_str(), laser_scan_frame_.c_str(), ex.what());
        }
        if (count > 10)
        {
            RCLCPP_ERROR(this->get_logger(), "No transform found. Aborting...");
            exit(-1);
        }
    } while (rclcpp::ok() && !initialized_);

    RCLCPP_WARN(this->get_logger(), "tf from %s to %s OK", robot_frame_.c_str(), laser_scan_frame_.c_str());

    // =============================
    // ! WAITING FOR POINTCLOUD TRANSFORM
    // =============================
    count = 0;
    initialized_ = false;
    do
    {
        try
        {
            transform = tf_buffer_->lookupTransform(robot_frame_, point_cloud_frame_, tf2::TimePointZero, tf2::durationFromSec(1.0));
            initialized_ = true;
        }
        catch (tf2::TransformException &ex)
        {

            count++;
            RCLCPP_WARN(this->get_logger(), "Cannot find tf from %s to %s: %s",
                        robot_frame_.c_str(), point_cloud_frame_.c_str(), ex.what());
        }
        if (count > 10)
        {
            RCLCPP_ERROR(this->get_logger(), "No transform found. Aborting...");
            exit(-1);
        }
    } while (rclcpp::ok() && !initialized_);

    RCLCPP_WARN(this->get_logger(), "tf from %s to %s OK", robot_frame_.c_str(), point_cloud_frame_.c_str());

    //=======================================================================
    // Octree
    //=======================================================================
    if (offline_octomap_path_.size() > 0)
        octree_ = new octomap::OcTree(offline_octomap_path_);
    else
        octree_ = new octomap::OcTree(octree_resol_);
    RCLCPP_WARN(this->get_logger(), "Loaded octree");

    octree_->setProbHit(prob_hit_);
    octree_->setProbMiss(prob_miss_);
    octree_->setClampingThresMin(clamping_thres_min_);
    octree_->setClampingThresMax(clamping_thres_max_);
    // a stored map keeps the default threshold, it was built with it
    if (offline_octomap_path_.size() == 0)
        octree_->setOccupancyThres(occupancy_thres_);

    //=======================================================================
    // Scan insertion
    //=======================================================================

    // depth camera: obstacles above the floor margin and within its reliable range,
    // in voxels with enough points; every other valid point only clears
    point_cloud_params_.min_z = min_z_pc_ + pc_min_z_offset_;
    point_cloud_params_.max_z = max_z_pc_;
    point_cloud_params_.min_range = minimum_range_;
    point_cloud_params_.max_range = mapping_max_range_;
    point_cloud_params_.max_obstacle_range = pc_max_obstacle_range_;
    point_cloud_params_.min_points_per_voxel = static_cast<unsigned int>(std::max(pc_voxel_min_points_, 1));
    point_cloud_params_.agent_radius = agent_filter_radius_;
    point_cloud_params_.range_noise = pc_range_noise_;

    // laser scan: a single beam per voxel is enough
    laser_scan_params_ = point_cloud_params_;
    laser_scan_params_.min_z = min_z_pc_;
    laser_scan_params_.max_obstacle_range = -1.0;
    laser_scan_params_.min_points_per_voxel = 1;
    laser_scan_params_.range_noise = 0.0;

    scan_integrator_.setDecayTime(obstacle_decay_time_);

    //=======================================================================
    // Gridmap
    //=======================================================================

    initializeGridMap();

    // SOCIAL HEATMAP
    social_heatmap_.setTimeDecayFactor(social_heatmap_decay_factor_);

    //=======================================================================
    // Publishers
    //=======================================================================
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    octomap_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("octomap_map", 2);
    grid_map_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>("social_grid_map", 1);
    relevant_agents_pub_ = this->create_publisher<pedsim_msgs::msg::AgentStates>("relevant_agents", qos);

    //=======================================================================
    // Subscribers
    //=======================================================================

    // Agent states callback
    agent_states_sub_ = this->create_subscription<pedsim_msgs::msg::AgentStates>(
        social_agents_topic_, 1, std::bind(&WorldModeler::agentStatesCallback, this, std::placeholders::_1));

    // Odometry data (feedback)
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odometry_topic_, 1, std::bind(&WorldModeler::odomCallback, this, std::placeholders::_1));

    // Map erase command
    erase_map_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        erase_map_topic_, 1, std::bind(&WorldModeler::eraseMapCallback, this, std::placeholders::_1));

    nav_sts_available_ = false;
    if (!nav_sts_available_)
        RCLCPP_WARN(this->get_logger(), "Waiting for odometry.");

    rclcpp::Rate loop_rate(10);
    while (rclcpp::ok() && !nav_sts_available_)
    {
        rclcpp::spin_some(this->get_node_base_interface());
        loop_rate.sleep();
    }
    RCLCPP_WARN(this->get_logger(), "Odometry received.");

    if (offline_octomap_path_.size() == 0)
    {
        // POINTCLOUD
        std::vector<std::string> pc_need_frames;
        pc_need_frames.push_back(point_cloud_frame_);
        pc_need_frames.push_back(fixed_frame_);
        pc_need_frames.push_back(robot_frame_);
        point_cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, point_cloud_topic_);
        point_cloud_mn_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>>(
            *tf_buffer_,
            point_cloud_frame_,
            1,
            this->get_node_logging_interface(),
            this->get_node_clock_interface());
        point_cloud_mn_->connectInput(*point_cloud_sub_);
        point_cloud_mn_->setTargetFrames(pc_need_frames);
        point_cloud_mn_->registerCallback(&WorldModeler::pointCloudCallback, this);

        // LASERSCAN
        std::vector<std::string> laser_need_frames;
        laser_need_frames.push_back(point_cloud_frame_);
        laser_need_frames.push_back(fixed_frame_);
        laser_need_frames.push_back(robot_frame_);

        laser_scan_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::LaserScan>>(this, laser_scan_topic_);
        laser_scan_mn_ = std::make_shared<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>>(
            *tf_buffer_,
            laser_scan_frame_,
            1,
            this->get_node_logging_interface(),
            this->get_node_clock_interface());
        laser_scan_mn_->connectInput(*laser_scan_sub_);
        laser_scan_mn_->setTargetFrames(laser_need_frames);
        laser_scan_mn_->registerCallback(&WorldModeler::laserScanCallback, this);
    }

    //=======================================================================
    // Services
    //=======================================================================
    save_binary_octomap_srv_ = this->create_service<std_srvs::srv::Empty>(
        "save_binary", std::bind(&WorldModeler::saveBinaryOctomapSrv, this, std::placeholders::_1, std::placeholders::_2));
    save_full_octomap_srv_ = this->create_service<std_srvs::srv::Empty>(
        "save_full", std::bind(&WorldModeler::saveFullOctomapSrv, this, std::placeholders::_1, std::placeholders::_2));
    get_binary_octomap_srv_ = this->create_service<octomap_msgs::srv::GetOctomap>(
        "get_binary", std::bind(&WorldModeler::getBinaryOctomapSrv, this, std::placeholders::_1, std::placeholders::_2));
    get_grid_map_srv_ = this->create_service<grid_map_msgs::srv::GetGridMap>(
        "get_grid_map", std::bind(&WorldModeler::getGridMapSrv, this, std::placeholders::_1, std::placeholders::_2));

    // Timer for publishing
    if (rviz_timer_ > 0.0)
    {
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(rviz_timer_), std::bind(&WorldModeler::timerCallback, this));
    }
}

//! Destructor.
WorldModeler::~WorldModeler()
{
    RCLCPP_INFO(this->get_logger(), "Octree has been deleted.");
    delete octree_;
}

void WorldModeler::initializeGridMap()
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    grid_map_ = grid_map::GridMap();
    grid_map_.setFrameId(map_frame_);
    grid_map_.add("obstacle");
    grid_map_.add("obstacles");
    grid_map_.add("full");
    grid_map_.add("comfort");
    grid_map_.add("social_heatmap");
    grid_map_.setGeometry(grid_map::Length(1, 1), octree_resol_);
}

void WorldModeler::eraseMap()
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    octree_->clear();
    scan_integrator_.reset();
    initializeGridMap();

    social_heatmap_ = SocialHeatmap();
    social_heatmap_.setTimeDecayFactor(social_heatmap_decay_factor_);
    relevant_agent_states_.agent_states.clear();
    social_agents_in_radius_.agent_states.clear();
    social_agents_in_radius_vector_.clear();
    orientation_drift_ = 0.0;
    position_drift_ = 0.0;

    RCLCPP_INFO(this->get_logger(), "Map erased. Starting a fresh map.");
}

void WorldModeler::eraseMapCallback(const std_msgs::msg::Bool::SharedPtr erase_map_msg)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (!erase_map_msg->data)
    {
        return;
    }

    eraseMap();
    skip_sensor_callbacks_after_erase_ = true;
    skip_sensor_callbacks_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    RCLCPP_WARN(this->get_logger(), "Skipping sensor callbacks for 3 seconds after erase_map");
}

//! Laserscan callback.
/*!
 * Callback for receiving the laserscan data (taken from octomap_server)
 */
void WorldModeler::laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr laser_scan_msg)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (skip_sensor_callbacks_after_erase_)
    {
        if (std::chrono::steady_clock::now() < skip_sensor_callbacks_until_)
        {
            return;
        }

        skip_sensor_callbacks_after_erase_ = false;
        RCLCPP_WARN(this->get_logger(), "Sensor callbacks resumed after erase_map cooldown");
    }

    geometry_msgs::msg::TransformStamped tf_map_to_fixed;

    try
    {
        // Check drift
        tf_map_to_fixed = tf_buffer_->lookupTransform(map_frame_, fixed_frame_, tf2::TimePointZero);
        tf2::Matrix3x3 m_map_to_fixed(tf2::Quaternion(
            tf_map_to_fixed.transform.rotation.x,
            tf_map_to_fixed.transform.rotation.y,
            tf_map_to_fixed.transform.rotation.z,
            tf_map_to_fixed.transform.rotation.w));
        tf2::Vector3 p_map_to_fixed(
            tf_map_to_fixed.transform.translation.x,
            tf_map_to_fixed.transform.translation.y,
            tf_map_to_fixed.transform.translation.z);

        double roll, pitch, yaw;
        m_map_to_fixed.getRPY(roll, pitch, yaw);
        orientation_drift_ += std::abs(prev_map_to_fixed_yaw_ - yaw);
        prev_map_to_fixed_yaw_ = yaw;
        position_drift_ += std::sqrt(std::pow(prev_map_to_fixed_pos_.x() - p_map_to_fixed.x(), 2.0) +
                                     std::pow(prev_map_to_fixed_pos_.y() - p_map_to_fixed.y(), 2.0));
        prev_map_to_fixed_pos_ = p_map_to_fixed;

        if (orientation_drift_ > 0.05 || position_drift_ > 0.05)
        {
            orientation_drift_ = 0.0;
            position_drift_ = 0.0;
            octree_->clear();
            scan_integrator_.reset();
            // mergeGlobalMapToOctomap();
        }
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
        return;
    }

    // Editable message
    sensor_msgs::msg::LaserScan laser_scan = *laser_scan_msg;

    // Apply single outliers removal filter
    if (apply_filter_)
    {
        filterSingleOutliers(laser_scan);
    }

    geometry_msgs::msg::TransformStamped sensorToWorldTf;
    try
    {
        sensorToWorldTf = tf_buffer_->lookupTransform(fixed_frame_, laser_scan_msg->header.frame_id, laser_scan_msg->header.stamp);
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Transform error of sensor data: %s, quitting callback", ex.what());
        return;
    }

    tf2::Transform sensorToWorld;
    tf2::fromMsg(sensorToWorldTf.transform, sensorToWorld);

    // Project to (x,y,z) in the fixed frame. Readings without return (+inf or exactly range_max,
    // REP 117, and zero only for drivers that report no return that way) are never obstacles, with
    // add_max_ranges they clear the space in front of the sensor. Other readings out of
    // [range_min, range_max] are discarded, e.g. range_max + 1 written by the in-place filters of
    // laser_filters over the parts of the robot that block the laser.
    octomap::Pointcloud returns, no_returns;
    const double no_return_range = laser_scan.range_max * 0.95;

    for (std::size_t i = 0; i < laser_scan.ranges.size(); i++)
    {
        double range = laser_scan.ranges[i];
        const bool no_return = (std::isinf(range) && range > 0.0) || range == laser_scan.range_max ||
                               (laser_zero_is_no_return_ && range == 0.0);

        if (no_return)
        {
            if (!add_max_ranges_ || !std::isfinite(no_return_range))
                continue;
            range = no_return_range;
        }
        else if (!std::isfinite(range) || range < laser_scan.range_min || range > laser_scan.range_max)
        {
            continue;
        }

        const double angle = laser_scan.angle_min + i * laser_scan.angle_increment;
        const tf2::Vector3 point = sensorToWorld * tf2::Vector3(range * std::cos(angle), range * std::sin(angle), 0.0);

        if (no_return)
            no_returns.push_back(point.x(), point.y(), point.z());
        else
            returns.push_back(point.x(), point.y(), point.z());
    }

    const geometry_msgs::msg::Vector3 &sensor_origin = sensorToWorldTf.transform.translation;

    // the camera never clears the layer of the scan plane: the laser sees obstacles there
    // that the camera may miss (thin or dark legs), and they would never stay mapped
    point_cloud_params_.keep_min_z = sensor_origin.z - 0.45 * octree_->getResolution();
    point_cloud_params_.keep_max_z = sensor_origin.z + 0.45 * octree_->getResolution();

    scan_integrator_.insertScan(*octree_, octomap::point3d(sensor_origin.x, sensor_origin.y, sensor_origin.z),
                                returns, no_returns, getAgentPositions(laser_scan_msg->header.stamp),
                                laser_scan_params_, rclcpp::Time(laser_scan_msg->header.stamp).seconds());
}

//! Pointcloud callback.
/*!
 * Callback for receiving the pointcloud data (taken from octomap_server)
 */
void WorldModeler::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (skip_sensor_callbacks_after_erase_)
    {
        if (std::chrono::steady_clock::now() < skip_sensor_callbacks_until_)
        {
            return;
        }

        skip_sensor_callbacks_after_erase_ = false;
        RCLCPP_WARN(this->get_logger(), "Sensor callbacks resumed after erase_map cooldown");
    }

    geometry_msgs::msg::TransformStamped sensorToWorldTf;
    try
    {
        sensorToWorldTf = tf_buffer_->lookupTransform(fixed_frame_, cloud->header.frame_id, cloud->header.stamp);
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Transform error of sensor data: %s, quitting callback", ex.what());
        return;
    }

    Eigen::Matrix4f sensorToWorld;
    pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);

    PCLPointCloud pc; // input cloud
    pcl::fromROSMsg(*cloud, pc);

    // directly transform to map frame:
    pcl::transformPointCloud(pc, pc, sensorToWorld);

    // Every valid point is used: the ones that are not obstacles (floor, ceiling, far
    // points, people) still clear the space in front of them. Invalid depth (NaN) says
    // nothing about free space and is skipped.
    octomap::Pointcloud returns;
    returns.reserve(pc.size());
    for (const PCLPoint &point : pc)
    {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
        {
            returns.push_back(point.x, point.y, point.z);
        }
    }

    const geometry_msgs::msg::Vector3 &sensor_origin = sensorToWorldTf.transform.translation;
    scan_integrator_.insertScan(*octree_, octomap::point3d(sensor_origin.x, sensor_origin.y, sensor_origin.z),
                                returns, octomap::Pointcloud(), getAgentPositions(cloud->header.stamp),
                                point_cloud_params_, rclcpp::Time(cloud->header.stamp).seconds());

    defineSocialGridMap();
}

//! Agent positions.
/*!
 * Positions of the agents around the robot, in the fixed frame, at the time of a scan.
 * Their tf frames are used when available and the reported poses otherwise.
 */
std::vector<octomap::point3d> WorldModeler::getAgentPositions(const builtin_interfaces::msg::Time &stamp)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    std::vector<octomap::point3d> agent_positions;

    for (const pedsim_msgs::msg::AgentState &agent_state : social_agents_in_radius_.agent_states)
    {
        const std::string agent_frame = "agent_" + std::to_string(agent_state.id);
        geometry_msgs::msg::TransformStamped transform;

        try
        {
            transform = tf_buffer_->lookupTransform(fixed_frame_, agent_frame, stamp);
        }
        catch (tf2::TransformException &ex)
        {
            try
            {
                // the last tf of an agent can be arbitrarily old, then the reported pose is better
                transform = tf_buffer_->lookupTransform(fixed_frame_, agent_frame, tf2::TimePointZero);
                if ((rclcpp::Time(stamp) - rclcpp::Time(transform.header.stamp)).seconds() > 0.5)
                {
                    throw tf2::LookupException("the last tf of " + agent_frame + " is too old");
                }
            }
            catch (tf2::TransformException &ex)
            {
                // no tf for this agent, use the pose it was reported with
                try
                {
                    transform = tf_buffer_->lookupTransform(fixed_frame_, agent_states_frame_, tf2::TimePointZero);
                }
                catch (tf2::TransformException &ex)
                {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), steady_clock_, 5000,
                                         "Cannot locate %s, its points will be mapped: %s", agent_frame.c_str(), ex.what());
                    continue;
                }

                geometry_msgs::msg::Point position;
                tf2::doTransform(agent_state.pose.position, position, transform);
                agent_positions.emplace_back(position.x, position.y, position.z);
                continue;
            }
        }

        agent_positions.emplace_back(transform.transform.translation.x,
                                     transform.transform.translation.y,
                                     transform.transform.translation.z);
    }

    return agent_positions;
}

//! Odometry callback.
/*!
 * Callback for getting updated vehicle odometry.
 */
void WorldModeler::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (!nav_sts_available_)
        nav_sts_available_ = true;

    robot_odometry_ = odom_msg;
}

//! Agent States callback.
/*!
 * Callback for getting updated agent states.
 */
void WorldModeler::agentStatesCallback(const pedsim_msgs::msg::AgentStates::SharedPtr agent_states_msg)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (nav_sts_available_)
    {
        agent_states_ = agent_states_msg;

        std::vector<pedsim_msgs::msg::AgentState> agent_state_vector;

        social_agents_in_radius_vector_.clear();

        double robot_velocity =
            std::sqrt(std::pow(robot_odometry_->twist.twist.linear.x, 2) + std::pow(robot_odometry_->twist.twist.linear.y, 2));

        actual_fov_distance_ = robot_distance_view_max_ / robot_velocity_threshold_ * robot_velocity;

        if (actual_fov_distance_ < robot_distance_view_min_)
        {
            actual_fov_distance_ = robot_distance_view_min_;
        }
        else if (actual_fov_distance_ > robot_distance_view_max_)
        {
            actual_fov_distance_ = robot_distance_view_max_;
        }

        if (!agent_states_msg->header.frame_id.empty())
        {
            agent_states_frame_ = agent_states_msg->header.frame_id;
        }

        for (int i = 0; i < agent_states_msg->agent_states.size(); i++)
        {
            // agents close enough to be seen by the sensors are filtered out of the map,
            // whether or not they are relevant for the planner
            double d_robot_agent = std::sqrt(std::pow(agent_states_msg->agent_states[i].pose.position.x - robot_odometry_->pose.pose.position.x, 2) +
                                             std::pow(agent_states_msg->agent_states[i].pose.position.y - robot_odometry_->pose.pose.position.y, 2));

            if (d_robot_agent < mapping_max_range_ + 2)
            {
                social_agents_in_radius_vector_.push_back(agent_states_msg->agent_states[i]);
            }

            if (this->isAgentInRFOV(agent_states_msg->agent_states[i]))
            {
                agent_state_vector.push_back(agent_states_msg->agent_states[i]);
            }
        }

        relevant_agent_states_.agent_states.clear();
        relevant_agent_states_.agent_states = agent_state_vector;

        social_agents_in_radius_.agent_states = social_agents_in_radius_vector_;

        relevant_agent_states_.header.stamp = this->get_clock()->now();
        relevant_agent_states_.header.frame_id = map_frame_;
    }
    relevant_agents_pub_->publish(relevant_agent_states_);
}

bool WorldModeler::isAgentInRFOV(const pedsim_msgs::msg::AgentState agent_state)
{
    if (!social_relevance_validity_checking_)
    {
        return true;
    }

    double d_robot_agent = std::sqrt(std::pow(agent_state.pose.position.x - robot_odometry_->pose.pose.position.x, 2) +
                                     std::pow(agent_state.pose.position.y - robot_odometry_->pose.pose.position.y, 2));

    if (d_robot_agent > actual_fov_distance_)
    {
        return false;
    }

    double tetha_robot_agent = atan2((agent_state.pose.position.y - robot_odometry_->pose.pose.position.y),
                                     (agent_state.pose.position.x - robot_odometry_->pose.pose.position.x));

    if (tetha_robot_agent < 0)
    {
        tetha_robot_agent = 2 * M_PI + tetha_robot_agent;
    }

    tf2::Quaternion q(robot_odometry_->pose.pose.orientation.x, robot_odometry_->pose.pose.orientation.y,
                      robot_odometry_->pose.pose.orientation.z, robot_odometry_->pose.pose.orientation.w);

    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    double robot_angle = yaw;

    if (robot_angle < 0)
    {
        robot_angle = 2 * M_PI + robot_angle;
    }

    if (tetha_robot_agent > (robot_angle + M_PI))
        tetha_robot_agent = abs(robot_angle + 2 * M_PI - tetha_robot_agent);
    else if (robot_angle > (tetha_robot_agent + M_PI))
        tetha_robot_agent = abs(tetha_robot_agent + 2 * M_PI - robot_angle);
    else
        tetha_robot_agent = abs(tetha_robot_agent - robot_angle);

    return abs(tetha_robot_agent) < robot_angle_view_;
}

bool WorldModeler::isRobotInFront(const pedsim_msgs::msg::AgentState agent_state, const grid_map::Position position)
{
    double tetha_agent_robot = atan2((position[1] - agent_state.pose.position.y),
                                     (position[0] - agent_state.pose.position.x));
    if (tetha_agent_robot < 0)
    {
        tetha_agent_robot = 2 * M_PI + tetha_agent_robot;
    }

    tf2::Quaternion q(agent_state.pose.orientation.x, agent_state.pose.orientation.y,
                      agent_state.pose.orientation.z, agent_state.pose.orientation.w);

    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    double agent_angle = yaw;

    if (agent_angle < 0)
    {
        agent_angle = 2 * M_PI + agent_angle;
    }

    if (tetha_agent_robot > (agent_angle + M_PI))
        tetha_agent_robot = abs(agent_angle + 2 * M_PI - tetha_agent_robot);
    else if (agent_angle > (tetha_agent_robot + M_PI))
        tetha_agent_robot = abs(tetha_agent_robot + 2 * M_PI - agent_angle);
    else
        tetha_agent_robot = abs(tetha_agent_robot - agent_angle);

    if (abs(tetha_agent_robot) < robot_angle_view_)
        return true;

    return false;
}

// !CALCULATE COMFORT AT AN SPECIFIC POSITION IN THE GRIDMAP

double WorldModeler::getExtendedPersonalSpace(const pedsim_msgs::msg::AgentState agent_state, const grid_map::Position position)
{
    double distance_robot_agent = std::sqrt(std::pow(agent_state.pose.position.x - position[0], 2) +
                                            std::pow(agent_state.pose.position.y - position[1], 2));

    double tetha_robot_agent = atan2((position[1] - agent_state.pose.position.y),
                                     (position[0] - agent_state.pose.position.x));

    if (tetha_robot_agent < 0)
    {
        tetha_robot_agent = 2 * M_PI + tetha_robot_agent;
    }

    double tetha_orientation;
    if (abs(agent_state.twist.linear.x) > 0 || abs(agent_state.twist.linear.y) > 0)
    {
        tetha_orientation = atan2(agent_state.twist.linear.y, agent_state.twist.linear.x);
    }
    else
    {
        tf2::Quaternion q(agent_state.pose.orientation.x, agent_state.pose.orientation.y,
                          agent_state.pose.orientation.z, agent_state.pose.orientation.w);

        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        tetha_orientation = yaw;
    }

    if (tetha_orientation < 0)
    {
        tetha_orientation = 2 * M_PI + tetha_orientation;
    }

    bool robot_in_front = false;
    bool robot_in_fov = false;
    double mod_sigma_y;
    double agent_velocity;

    agent_velocity =
        std::sqrt(std::pow(agent_state.twist.linear.x, 2) + std::pow(agent_state.twist.linear.y, 2));

    robot_in_front = this->isRobotInFront(agent_state, position);

    if (robot_in_front)
    {
        if (robot_in_fov)
            mod_sigma_y = (1 + agent_velocity * fv + fFront + fFieldOfView) * sigma_y;
        else
            mod_sigma_y = (1 + agent_velocity * fv + fFront) * sigma_y;
    }
    else
    {
        mod_sigma_y = sigma_y;
    }

    double basic_personal_space_value =
        social_comfort_amplitude_ *
        std::exp(-(
            std::pow(distance_robot_agent * std::sin(tetha_robot_agent - tetha_orientation) / (std::sqrt(2) * sigma_x),
                     2) +
            std::pow(distance_robot_agent * std::cos(tetha_robot_agent - tetha_orientation) / (std::sqrt(2) * mod_sigma_y),
                     2)));

    return basic_personal_space_value;
}

//! Time callback.
/*!
 * Callback for publishing the map periodically using the WorldModeler RViz plugin.
 */
void WorldModeler::timerCallback()
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    if (offline_octomap_path_.size() != 0)
    {
        defineSocialGridMap();
    }

    // Declare message
    octomap_msgs::msg::Octomap msg;
    octomap_msgs::binaryMapToMsg(*octree_, msg);
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = fixed_frame_;
    if (visualize_free_space_)
    {
        publishMap();

        grid_map_.setTimestamp(this->get_clock()->now().nanoseconds());
        std::shared_ptr<grid_map_msgs::msg::GridMap> message;
        message = grid_map::GridMapRosConverter::toMessage(grid_map_);
        grid_map_pub_->publish(*message);
    }
}

//! Save binary service
/*!
 * Service for saving the binary Octomap into the home folder
 */
bool WorldModeler::saveBinaryOctomapSrv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> req,
    std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    // Saves current octree_ in home folder
    std::string fpath(getenv("HOME"));
    octree_->writeBinary(fpath + "/map_laser_octomap.bt");
    return true;
}

//! Save binary service
/*!
 * Service for saving the full Octomap into the home folder
 */
bool WorldModeler::saveFullOctomapSrv(
    const std::shared_ptr<std_srvs::srv::Empty::Request> req,
    std::shared_ptr<std_srvs::srv::Empty::Response> res)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    // Saves current octree_ in home folder (full probabilities)
    std::string fpath(getenv("HOME"));
    octree_->write(fpath + "/map_laser_octomap.ot");
    return true;
}

//! Get binary service
/*!
 * Service for getting the binary Octomap
 */
bool WorldModeler::getBinaryOctomapSrv(
    const std::shared_ptr<OctomapSrv::Request> req,
    std::shared_ptr<OctomapSrv::GetOctomap::Response> res)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    RCLCPP_INFO(get_logger(), "Sending binary map data on service request");

    res->map.header.frame_id = fixed_frame_;
    res->map.header.stamp = this->get_clock()->now();

    if (!octomap_msgs::binaryMapToMsg(*octree_, res->map))
        return false;

    return true;
}

//! Get binary service
/*!
 * Service for getting the binary Octomap
 */
bool WorldModeler::getGridMapSrv(
    const std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> req,
    std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> res)
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    RCLCPP_INFO(get_logger(), "Sending grid map data on service");

    grid_map_.setTimestamp(this->get_clock()->now().nanoseconds());

    res->map = *grid_map::GridMapRosConverter::toMessage(grid_map_);

    return true;
}

//! Publish the map
/*!
 * Service for saving the binary of the Octomap into the home folder
 */
void WorldModeler::publishMap()
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    // Declare message and resize
    visualization_msgs::msg::MarkerArray occupiedNodesVis;
    occupiedNodesVis.markers.resize(octree_->getTreeDepth() + 1);

    // Octree limits for height map
    double min_x, min_y, max_x, max_y, min_z_, max_z_;
    octree_->getMetricMin(min_x, min_y, min_z_);
    octree_->getMetricMax(max_x, max_y, max_z_);

    // Traverse all leafs in the tree:
    for (octomap::OcTree::iterator it = octree_->begin(octree_->getTreeDepth()),
                                   end = octree_->end();
         it != end; ++it)
    {
        if (octree_->isNodeOccupied(*it))
        {
            double x = it.getX();
            double y = it.getY();
            double z = it.getZ();

            // Create marker:
            unsigned idx = it.getDepth();
            assert(idx < occupiedNodesVis.markers.size());

            geometry_msgs::msg::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
        }

        if (!rclcpp::ok())
            break;
    }
    // Finish Headers and options
    for (unsigned i = 0; i < occupiedNodesVis.markers.size(); ++i)
    {
        double size = octree_->getNodeSize(i);

        occupiedNodesVis.markers[i].header.frame_id = fixed_frame_;
        occupiedNodesVis.markers[i].header.stamp = this->get_clock()->now();
        occupiedNodesVis.markers[i].ns = fixed_frame_;
        occupiedNodesVis.markers[i].id = i;
        occupiedNodesVis.markers[i].type = visualization_msgs::msg::Marker::CUBE_LIST;
        occupiedNodesVis.markers[i].scale.x = size;
        occupiedNodesVis.markers[i].scale.y = size;
        occupiedNodesVis.markers[i].scale.z = size;
        occupiedNodesVis.markers[i].color.r = 0.5;
        occupiedNodesVis.markers[i].color.g = 0.5;
        occupiedNodesVis.markers[i].color.b = 0.5;
        occupiedNodesVis.markers[i].color.a = 1.0;

        if (occupiedNodesVis.markers[i].points.size() > 0)
            occupiedNodesVis.markers[i].action = visualization_msgs::msg::Marker::ADD;
        else
            occupiedNodesVis.markers[i].action = visualization_msgs::msg::Marker::DELETE;

        if (!rclcpp::ok())
            break;
    }

    // Publish it
    octomap_marker_pub_->publish(occupiedNodesVis);
}

void WorldModeler::filterSingleOutliers(sensor_msgs::msg::LaserScan &laser_scan_msg)
{
    // compare against the original readings, and drop outliers as NaN so they are
    // neither obstacles nor max range readings
    const std::vector<float> ranges = laser_scan_msg.ranges;
    int n(ranges.size());
    double thres(laser_scan_msg.range_max / 10.0);
    for (int i = 1; i < n - 1; i++)
    {
        if ((std::abs(ranges[i - 1] - ranges[i]) > thres) &&
            (std::abs(ranges[i + 1] - ranges[i]) > thres))
        {
            laser_scan_msg.ranges[i] = std::numeric_limits<float>::quiet_NaN();
        }

        if (!rclcpp::ok())
            break;
    }
}

void WorldModeler::defineSocialGridMap()
{
    std::lock_guard<std::recursive_mutex> lock(map_state_mutex_);

    // ! OCTOMAP PREPARATION

    grid_map::Position3 min_bound;
    grid_map::Position3 max_bound;

    grid_map_.clearAll();

    if (octree_->size() == 0)
    {
        RCLCPP_WARN(this->get_logger(), "Empty octree, keeping blank social grid map");
        return;
    }

    octree_->getMetricMin(min_bound(0), min_bound(1), min_bound(2));
    octree_->getMetricMax(max_bound(0), max_bound(1), max_bound(2));

    const double pad = octree_->getResolution() * 5;

    min_bound(0) -= pad;
    min_bound(1) -= pad;
    max_bound(0) += pad;
    max_bound(1) += pad;

    grid_map::GridMapOctomapConverter::fromOctomap(*octree_, "obstacles", grid_map_, &min_bound, &max_bound);
    if (!grid_map_.exists("obstacles"))
    {
        RCLCPP_WARN(this->get_logger(), "No obstacles layer after fromOctomap, skipping defineSocialGridMap");
        return;
    }
    if (!grid_map_.exists("full"))
    {
        grid_map_.add("full");
    }
    if (!grid_map_.exists("comfort"))
    {
        grid_map_.add("comfort");
    }
    if (!grid_map_.exists("social_heatmap"))
    {
        grid_map_.add("social_heatmap");
    }

    grid_map_["obstacles"] = 150 * grid_map_["obstacles"];
    grid_map_["full"] = 1.0 * grid_map_["obstacles"];

    grid_map::Matrix &full_grid_map = grid_map_["full"];
    grid_map::Matrix &comfort_grid_map = grid_map_["comfort"];

    // !SOCIAL AGENTS GRID MAP PREPARATION

    for (int i = 0; i < relevant_agent_states_.agent_states.size(); i++)
    {
        grid_map::Position center(relevant_agent_states_.agent_states[i].pose.position.x, relevant_agent_states_.agent_states[i].pose.position.y);

        for (grid_map::CircleIterator iterator(grid_map_, center, social_agent_radius_);
             !iterator.isPastEnd(); ++iterator)
        {
            try
            {
                grid_map::Index index(*iterator);
                full_grid_map(index(0), index(1)) = 100;
            }
            catch (const std::out_of_range &oor)
            {
                RCLCPP_ERROR(this->get_logger(), "TRIED TO DEFINE AN AGENT OUT OF RANGE");
            }
        }

        for (grid_map::CircleIterator iterator(grid_map_, center, 2.25);
             !iterator.isPastEnd(); ++iterator)
        {
            try
            {

                grid_map::Position temp_pos;

                grid_map_.getPosition(*iterator, temp_pos);

                grid_map::Index index(*iterator);

                double last_val = comfort_grid_map(index(0), index(1));

                if (std::isnan(last_val))
                {
                    last_val = getExtendedPersonalSpace(relevant_agent_states_.agent_states[i], temp_pos);
                }
                else
                {
                    last_val += getExtendedPersonalSpace(relevant_agent_states_.agent_states[i], temp_pos);
                }

                comfort_grid_map(index(0), index(1)) = last_val;
            }
            catch (const std::out_of_range &oor)
            {
                RCLCPP_ERROR(this->get_logger(), "TRIED TO DEFINE COMFORT OUT OF RANGE");
            }
        }
    }

    social_heatmap_.updateSocialHeatmap(grid_map_, relevant_agent_states_);

    grid_map_["full"] = full_grid_map;
    grid_map_["comfort"] = comfort_grid_map;
    grid_map_["social_heatmap"] = social_heatmap_.getSocialHeatmap();
}

//! Main function
int main(int argc, char **argv)
{
    //=======================================================================
    // Override SIGINT handler
    //=======================================================================
    signal(SIGINT, stopNode);

    // Init ROS node
    rclcpp::init(argc, argv);

    // Spin
    rclcpp::spin(std::make_shared<WorldModeler>());

    rclcpp::shutdown();

    // Exit main function without errors
    return 0;
}
