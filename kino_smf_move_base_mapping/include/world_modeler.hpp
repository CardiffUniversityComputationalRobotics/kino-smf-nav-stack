#ifndef WORLD_MODELER_HPP
#define WORLD_MODELER_HPP

#include <chrono>
#include <cmath>
#include <mutex>

// SOCIAL HEATMAP
#include "social_heatmap.hpp"

// OCTOMAP SCAN INSERTION (marking, free space clearing, agents and noise filtering)
#include "scan_integrator.hpp"

// ROS2
#include <rclcpp/rclcpp.hpp>

// ROS2 messages
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/int8.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

// ROS2 services
#include <std_srvs/srv/empty.hpp>

// ROS2 LaserScan tools
#include <laser_geometry/laser_geometry.hpp>

// tf2
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/message_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include "message_filters/subscriber.h"
#include <tf2_ros/create_timer_ros.h>

// Octomap
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/srv/get_octomap.hpp>
typedef octomap_msgs::srv::GetOctomap OctomapSrv;
#include <octomap/Pointcloud.h>

// grid map library
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_octomap/grid_map_octomap.hpp>
#include <grid_map_msgs/srv/get_grid_map.hpp>
#include <grid_map_cv/grid_map_cv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/crop_box.h>

// PEDSIM
#include <pedsim_msgs/msg/agent_states.hpp>

#include <signal.h>

typedef pcl::PointXYZ PCLPoint;
typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
typedef octomap::OcTree OcTreeT;

void stopNode(int sig)
{
    rclcpp::shutdown();
    exit(0);
}

//!  WorldModeler class.
/*!
 * Autopilot Laser WorldModeler.
 * Create a WorldModeler using information from laser scans.
 */
class WorldModeler : public rclcpp::Node
{
public:
    //! Constructor
    WorldModeler();
    //! Destructor
    virtual ~WorldModeler();
    //! Callback for getting the laser_scan data
    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr laser_scan_msg);
    //! Callback for getting the point_cloud data
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud);
    //! Callback for getting current vehicle odometry
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
    //! Callback for getting current agent states
    void agentStatesCallback(const pedsim_msgs::msg::AgentStates::SharedPtr agent_states_msg);
    //! Callback for erasing the current map
    void eraseMapCallback(const std_msgs::msg::Bool::SharedPtr erase_map_msg);
    bool isAgentInRFOV(const pedsim_msgs::msg::AgentState agent_state);
    //! Check if robot is in front of agent.
    bool isRobotInFront(pedsim_msgs::msg::AgentState agent_state, grid_map::Position position);
    //! Calculate comfort in specific position in gridmap
    double getExtendedPersonalSpace(pedsim_msgs::msg::AgentState agent_state, grid_map::Position position);
    //! Periodic callback to publish the map for visualization.
    void timerCallback();
    //! Positions (in the fixed frame) of the agents whose returns must not be mapped
    std::vector<octomap::point3d> getAgentPositions(const builtin_interfaces::msg::Time &stamp);
    //! Service to save a binary Octomap (.bt)
    bool saveBinaryOctomapSrv(const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                              std::shared_ptr<std_srvs::srv::Empty::Response> res);
    //! Service to save a full Octomap (.ot)
    bool saveFullOctomapSrv(const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                            std::shared_ptr<std_srvs::srv::Empty::Response> res);
    //! Service to get a binary Octomap
    bool getBinaryOctomapSrv(const std::shared_ptr<OctomapSrv::Request> req,
                             std::shared_ptr<OctomapSrv::Response> res);
    //! Service to get grid map
    bool getGridMapSrv(const std::shared_ptr<grid_map_msgs::srv::GetGridMap::Request> req,
                       std::shared_ptr<grid_map_msgs::srv::GetGridMap::Response> res);

    //! Publish the WorldModeler
    void publishMap();
    //! Redefine the gridmap with octomap
    void defineSocialGridMap();

private:
    //! Reset map data so mapping can start again from an empty state.
    void eraseMap();

    //! Recreate a blank grid map with the expected layers.
    void initializeGridMap();

    //! Filter outliers
    void filterSingleOutliers(sensor_msgs::msg::LaserScan &laser_scan_msg);

    // ROS2
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr octomap_marker_pub_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;
    rclcpp::Publisher<pedsim_msgs::msg::AgentStates>::SharedPtr relevant_agents_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<pedsim_msgs::msg::AgentStates>::SharedPtr agent_states_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr erase_map_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> point_cloud_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> laser_scan_sub_;

    // SERVICES
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr save_binary_octomap_srv_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr save_full_octomap_srv_;
    rclcpp::Service<OctomapSrv>::SharedPtr get_binary_octomap_srv_;
    rclcpp::Service<grid_map_msgs::srv::GetGridMap>::SharedPtr get_grid_map_srv_;

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> point_cloud_mn_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>> laser_scan_mn_;

    // tf2
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    // Names
    std::string map_frame_, fixed_frame_, robot_frame_, offline_octomap_path_,
        odometry_topic_, social_agents_topic_, erase_map_topic_;

    // Laser scans
    std::string laser_scan_frame_, laser_scan_topic_;

    // Point Clouds
    std::string point_cloud_topic_, point_cloud_frame_;

    nav_msgs::msg::Odometry::SharedPtr robot_odometry_;

    SocialHeatmap social_heatmap_ = SocialHeatmap();

    // pedsim messages
    pedsim_msgs::msg::AgentStates::SharedPtr agent_states_;
    pedsim_msgs::msg::AgentStates relevant_agent_states_;

    pedsim_msgs::msg::AgentStates social_agents_in_radius_;
    std::vector<pedsim_msgs::msg::AgentState> social_agents_in_radius_vector_;
    std::string agent_states_frame_;

    double social_agent_radius_;

    // Octree
    octomap::OcTree *octree_;
    double octree_resol_, minimum_range_, rviz_timer_;
    double mapping_max_range_, min_z_pc_, max_z_pc_;

    // Octree sensor model (2 hits to become an obstacle)
    double prob_hit_ = 0.7;
    double prob_miss_ = 0.4;
    double clamping_thres_min_ = 0.1192;
    double clamping_thres_max_ = 0.971;
    double occupancy_thres_ = 0.75;

    // Scan insertion: free space clearing, noise and agents filtering (defaults for a D435 at 848x480)
    ScanIntegrator scan_integrator_;
    ScanIntegrator::Params point_cloud_params_, laser_scan_params_;
    double pc_min_z_offset_ = 0.2;
    double pc_max_obstacle_range_ = 3.5;
    int pc_voxel_min_points_ = 3;
    double pc_range_noise_ = 0.004;
    double agent_filter_radius_ = 0.6;
    double obstacle_decay_time_ = 0.0;
    bool laser_zero_is_no_return_ = false;

    //! for throttled warnings, the node clock may not run (use_sim_time without /clock)
    rclcpp::Clock steady_clock_{RCL_STEADY_TIME};

    // grid map
    grid_map::GridMap grid_map_;

    grid_map_msgs::msg::GridMap grid_map_msg_;

    std::recursive_mutex map_state_mutex_;

    // social relevance validity checking constants
    double robot_distance_view_max_, robot_distance_view_min_, robot_velocity_threshold_, robot_angle_view_, actual_fov_distance_;
    double social_heatmap_decay_factor_;

    // Basic social personal space parameters defined
    double social_comfort_amplitude_ = 4;
    double sigma_x = 0.45;
    double sigma_y = 0.45;
    double fv = 0.8;
    double fFront = 0.2;
    double fFieldOfView = 0.0;

    // Flags
    tf2::Vector3 prev_map_to_fixed_pos_{0.0, 0.0, 0.0};
    bool initialized_;
    bool nav_sts_available_;
    bool visualize_free_space_;
    bool social_relevance_validity_checking_;

    bool apply_filter_;
    bool add_max_ranges_;
    bool add_rays_;
    bool skip_sensor_callbacks_after_erase_;

    double orientation_drift_, prev_map_to_fixed_yaw_ = 0.0, position_drift_;
    std::chrono::steady_clock::time_point skip_sensor_callbacks_until_;
};

#endif // WORLD_MODELER_HPP
