#ifndef SCAN_INTEGRATOR_HPP
#define SCAN_INTEGRATOR_HPP

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <octomap/octomap.h>

//!  ScanIntegrator class.
/*!
 * Inserts range measurements in an OcTree so that the map stays consistent in a
 * dynamic, human populated scene:
 *  - every valid return carves free space along its ray, also the returns that are
 *    not mapped (floor, ceiling, far points, people), so voxels left behind by people
 *    or by noise are erased as soon as the sensor sees through them;
 *  - returns are grouped per voxel: one ray is cast per voxel, and a voxel is only
 *    marked when it gathers enough returns, which rejects speckle and flying pixels;
 *  - returns close to the tracked agents only clear, they are never mapped;
 *  - optionally, occupied voxels that are not observed again are cleared after a while.
 * A sensor only updates the voxels of its own height band, so it never erases what
 * another sensor maps at heights it does not map itself.
 */
class ScanIntegrator
{
public:
    //! Insertion parameters of a sensor (distances in meters, heights in the octree frame)
    struct Params
    {
        //! the sensor maps obstacles and clears free space only between these heights
        double min_z = std::numeric_limits<double>::lowest();
        double max_z = std::numeric_limits<double>::max();
        //! returns closer than this are ignored (< 0 disables)
        double min_range = -1.0;
        //! rays are truncated at this range, farther returns only clear (< 0 disables)
        double max_range = -1.0;
        //! returns farther than this only clear (< 0 uses max_range)
        double max_obstacle_range = -1.0;
        //! returns a voxel needs in a single scan to be marked as occupied
        unsigned int min_points_per_voxel = 1;
        //! returns closer than this to an agent (in x-y) only clear
        double agent_radius = 0.0;
        //! range error of the sensor, sigma = range_noise * range^2: clearing rays stop
        //! 3 sigma before their return so noisy points do not erase the surface they hit
        double range_noise = 0.0;
        //! voxels between these heights are never cleared by this sensor, e.g. the layer of a
        //! laser scan, which may see obstacles there that this sensor misses (unset if min > max)
        double keep_min_z = 0.0;
        double keep_max_z = -1.0;
    };

    //! CONSTRUCTOR

    ScanIntegrator();

    //! FUNCTIONS

    //! Insert a scan taken from origin. returns are the measured points and no_returns the
    //! end of rays that did not hit anything (e.g. laser max range readings), all in the
    //! octree frame. agents are the positions of the people that must not be mapped.
    void insertScan(octomap::OcTree &octree, const octomap::point3d &origin,
                    const octomap::Pointcloud &returns, const octomap::Pointcloud &no_returns,
                    const std::vector<octomap::point3d> &agents, const Params &params, double stamp);

    //! Forget what is known about the octree, needed whenever the octree is cleared.
    void reset();

    //! SETTERS

    //! Occupied voxels that are not observed for decay_time seconds are cleared (<= 0 disables).
    void setDecayTime(double decay_time);

private:
    struct VoxelReturns
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        unsigned int count = 0;
    };
    typedef std::unordered_map<octomap::OcTreeKey, VoxelReturns, octomap::OcTreeKey::KeyHash> VoxelReturnsMap;

    void addReturn(const octomap::OcTree &octree, const octomap::point3d &point, VoxelReturnsMap &voxels) const;

    bool isNearAgent(const octomap::point3d &point, const std::vector<octomap::point3d> &agents, double radius) const;

    //! Prepare the scratch grid that covers the origin and the voxels of the current scan,
    //! between the heights of the sensor.
    bool resetScanGrid(const octomap::OcTree &octree, const octomap::point3d &origin, const Params &params);

    //! Index of a key in the scratch grid, or -1 when it is outside.
    long scanGridIndex(const octomap::OcTreeKey &key) const;

    //! Collect the voxels traversed by the ray inside the scan heights as free.
    void castRay(const octomap::OcTree &octree, const octomap::point3d &origin, const octomap::point3d &end,
                 bool include_end);

    void markFree(const octomap::OcTreeKey &key);

    void decay(octomap::OcTree &octree, double stamp);

    double decay_time_ = 0.0;
    double last_decay_stamp_ = 0.0;

    //! last time each occupied voxel was observed (only kept when decay is enabled)
    std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> last_hit_stamps_;

    // per scan storage, kept to reuse the memory
    VoxelReturnsMap hit_voxels_, free_voxels_;
    std::vector<octomap::OcTreeKey> free_cells_, occupied_cells_;
    octomap::KeyRay key_ray_; // temp storage for ray casting

    // Dense grid around the sensor used to find the cells of a scan without hashing:
    // a cell holds 2 * scan_id_ when it was collected as free in the current scan and
    // 2 * scan_id_ + 1 when a return fell inside it (those are never cleared).
    std::vector<uint32_t> scan_grid_;
    uint32_t scan_id_ = 0;
    octomap::OcTreeKey scan_grid_min_;
    long scan_grid_size_[3] = {0, 0, 0};
    //! rays are clipped between these heights (faces of the voxels at the sensor heights)
    double scan_min_z_ = -std::numeric_limits<double>::infinity();
    double scan_max_z_ = std::numeric_limits<double>::infinity();
    //! height keys of the voxels the current scan must not clear
    long scan_keep_min_key_ = 1;
    long scan_keep_max_key_ = 0;
};

#endif // SCAN_INTEGRATOR_HPP
