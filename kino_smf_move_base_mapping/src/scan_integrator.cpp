#include "scan_integrator.hpp"

#include <algorithm>
#include <cmath>

//! CONSTRUCTOR
ScanIntegrator::ScanIntegrator()
{
}

//! FUNCTIONS
void ScanIntegrator::insertScan(octomap::OcTree &octree, const octomap::point3d &origin,
                                const octomap::Pointcloud &returns, const octomap::Pointcloud &no_returns,
                                const std::vector<octomap::point3d> &agents, const Params &params, double stamp)
{
    if (!std::isfinite(origin.x()) || !std::isfinite(origin.y()) || !std::isfinite(origin.z()))
    {
        return;
    }

    // Group the endpoints per voxel. A single ray per voxel carves the same free space
    // as one ray per point (a depth image has hundreds of points per voxel), and the
    // number of returns in a voxel tells real surfaces from isolated noise.
    hit_voxels_.clear();
    free_voxels_.clear();

    // a return beyond max_range is not trusted, but the ray is free up to max_range,
    // minus the range error of such a return
    const double range_noise = std::max(params.range_noise, 0.0);
    const double free_range = params.max_range - 3.0 * range_noise * params.max_range * params.max_range;

    for (const octomap::point3d &point : returns)
    {
        const double range = (point - origin).norm();

        // invalid depth may come as points at the sensor origin
        if (range < 1e-3 || (params.min_range >= 0.0 && range < params.min_range))
        {
            continue;
        }

        if (params.max_range >= 0.0 && range > params.max_range)
        {
            if (free_range > 0.0)
            {
                addReturn(octree, origin + (point - origin) * (free_range / range), free_voxels_);
            }
            continue;
        }

        addReturn(octree, point, hit_voxels_);
    }

    for (const octomap::point3d &point : no_returns)
    {
        const double range = (point - origin).norm();

        if (params.max_range >= 0.0 && range > params.max_range)
        {
            addReturn(octree, origin + (point - origin) * (params.max_range / range), free_voxels_);
        }
        else
        {
            addReturn(octree, point, free_voxels_);
        }
    }

    if (!resetScanGrid(octree, origin, params))
    {
        return;
    }

    // a voxel with a return of this scan is never cleared by it, even when it is not
    // marked (e.g. a person, or a thin obstacle with few returns)
    for (const auto &voxel : hit_voxels_)
    {
        const long index = scanGridIndex(voxel.first);
        if (index >= 0)
        {
            scan_grid_[index] = 2 * scan_id_ + 1;
        }
    }

    const double max_obstacle_range = params.max_obstacle_range >= 0.0 ? params.max_obstacle_range : params.max_range;

    free_cells_.clear();
    occupied_cells_.clear();

    for (const auto &voxel : hit_voxels_)
    {
        const VoxelReturns &voxel_returns = voxel.second;
        const octomap::point3d end(voxel_returns.x / voxel_returns.count,
                                   voxel_returns.y / voxel_returns.count,
                                   voxel_returns.z / voxel_returns.count);
        const double range = (end - origin).norm();

        // free cells, up to 3 sigma before the return
        const double margin = 3.0 * range_noise * range * range;
        if (range > margin)
        {
            castRay(octree, origin, origin + (end - origin) * ((range - margin) / range), false);
        }

        // occupied endpoint: only voxels with enough returns, inside the obstacle height
        // band and range, and away from people. The rest of the returns (floor, ceiling,
        // far or noisy points, people) only clear the space in front of them.
        if (voxel_returns.count >= params.min_points_per_voxel &&
            end.z() >= params.min_z && end.z() <= params.max_z &&
            (max_obstacle_range < 0.0 || range <= max_obstacle_range) &&
            !(params.agent_radius > 0.0 && isNearAgent(end, agents, params.agent_radius)) &&
            scanGridIndex(voxel.first) >= 0)
        {
            occupied_cells_.push_back(voxel.first);
        }
    }

    for (const auto &voxel : free_voxels_)
    {
        const VoxelReturns &voxel_returns = voxel.second;
        const octomap::point3d end(voxel_returns.x / voxel_returns.count,
                                   voxel_returns.y / voxel_returns.count,
                                   voxel_returns.z / voxel_returns.count);

        // nothing was hit: the end of the ray is free too
        castRay(octree, origin, end, true);
    }

    for (const octomap::OcTreeKey &key : free_cells_)
    {
        octree.updateNode(key, false);
    }

    // now mark all occupied cells
    for (const octomap::OcTreeKey &key : occupied_cells_)
    {
        octree.updateNode(key, true);

        if (decay_time_ > 0.0)
        {
            last_hit_stamps_[key] = stamp;
        }
    }

    if (decay_time_ > 0.0)
    {
        decay(octree, stamp);
    }
}

void ScanIntegrator::reset()
{
    last_hit_stamps_.clear();
    last_decay_stamp_ = 0.0;
}

void ScanIntegrator::addReturn(const octomap::OcTree &octree, const octomap::point3d &point, VoxelReturnsMap &voxels) const
{
    octomap::OcTreeKey key;
    if (!std::isfinite(point.x()) || !std::isfinite(point.y()) || !std::isfinite(point.z()) ||
        !octree.coordToKeyChecked(point, key))
    {
        return;
    }

    VoxelReturns &voxel_returns = voxels[key];
    voxel_returns.x += point.x();
    voxel_returns.y += point.y();
    voxel_returns.z += point.z();
    voxel_returns.count++;
}

bool ScanIntegrator::isNearAgent(const octomap::point3d &point, const std::vector<octomap::point3d> &agents, double radius) const
{
    for (const octomap::point3d &agent : agents)
    {
        const double dx = point.x() - agent.x();
        const double dy = point.y() - agent.y();
        if (dx * dx + dy * dy <= radius * radius)
        {
            return true;
        }
    }

    return false;
}

bool ScanIntegrator::resetScanGrid(const octomap::OcTree &octree, const octomap::point3d &origin, const Params &params)
{
    octomap::OcTreeKey origin_key;
    if (!octree.coordToKeyChecked(origin, origin_key))
    {
        return false;
    }

    // the grid spans the sensor and every voxel of the scan: since it is a box, it also
    // contains the rays between them
    long min_key[3], max_key[3];
    for (unsigned int i = 0; i < 3; ++i)
    {
        min_key[i] = max_key[i] = origin_key[i];
    }

    for (const VoxelReturnsMap *voxels : {&hit_voxels_, &free_voxels_})
    {
        for (const auto &voxel : *voxels)
        {
            for (unsigned int i = 0; i < 3; ++i)
            {
                min_key[i] = std::min(min_key[i], static_cast<long>(voxel.first[i]));
                max_key[i] = std::max(max_key[i], static_cast<long>(voxel.first[i]));
            }
        }
    }

    // but only between the heights of the sensor, nothing else is updated by it (heights
    // out of the octree, like the unset defaults, leave that side open)
    const double resolution = octree.getResolution();
    const double octree_limit = resolution * (1 << (octree.getTreeDepth() - 1));
    auto heightKey = [&](double z, octomap::key_type &key)
    { return std::abs(z) < octree_limit && octree.coordToKeyChecked(z, key); };

    scan_min_z_ = -std::numeric_limits<double>::infinity();
    scan_max_z_ = std::numeric_limits<double>::infinity();

    octomap::key_type z_key;
    if (heightKey(params.min_z, z_key))
    {
        min_key[2] = std::max(min_key[2], static_cast<long>(z_key));
        scan_min_z_ = octree.keyToCoord(z_key) - resolution / 2.0;
    }
    if (heightKey(params.max_z, z_key))
    {
        max_key[2] = std::min(max_key[2], static_cast<long>(z_key));
        scan_max_z_ = octree.keyToCoord(z_key) + resolution / 2.0;
    }

    scan_keep_min_key_ = 1;
    scan_keep_max_key_ = 0;
    octomap::key_type keep_min_key, keep_max_key;
    if (params.keep_min_z <= params.keep_max_z &&
        heightKey(params.keep_min_z, keep_min_key) && heightKey(params.keep_max_z, keep_max_key))
    {
        scan_keep_min_key_ = keep_min_key;
        scan_keep_max_key_ = keep_max_key;
    }

    // keep the grid bounded whatever the input (e.g. no max_range and a far outlier):
    // voxels out of it are neither marked nor cleared
    if (max_key[2] - min_key[2] + 1 > 256)
    {
        min_key[2] = std::max(min_key[2], static_cast<long>(origin_key[2]) - 128);
        max_key[2] = std::min(max_key[2], static_cast<long>(origin_key[2]) + 128);
    }

    if (min_key[2] > max_key[2])
    {
        return false;
    }

    // at most 2^22 cells (16 MB): about +-30 m around the sensor for a 1.2 m high band at 0.1 m
    const long max_half_size = static_cast<long>(std::sqrt((1 << 22) / static_cast<double>(max_key[2] - min_key[2] + 1))) / 2;
    for (unsigned int i = 0; i < 2; ++i)
    {
        min_key[i] = std::max(min_key[i], static_cast<long>(origin_key[i]) - max_half_size);
        max_key[i] = std::min(max_key[i], static_cast<long>(origin_key[i]) + max_half_size);
    }

    std::size_t cells = 1;
    for (unsigned int i = 0; i < 3; ++i)
    {
        scan_grid_min_[i] = static_cast<octomap::key_type>(min_key[i]);
        scan_grid_size_[i] = max_key[i] - min_key[i] + 1;
        cells *= static_cast<std::size_t>(scan_grid_size_[i]);
    }

    if (scan_grid_.size() < cells)
    {
        scan_grid_.resize(cells, 0);
    }

    // cells written by older scans hold smaller ids, so the grid is never cleared,
    // except before the ids overflow
    if (scan_id_ >= std::numeric_limits<uint32_t>::max() / 2 - 1)
    {
        std::fill(scan_grid_.begin(), scan_grid_.end(), 0);
        scan_id_ = 0;
    }
    scan_id_++;

    return true;
}

long ScanIntegrator::scanGridIndex(const octomap::OcTreeKey &key) const
{
    long index = 0;
    for (int i = 2; i >= 0; --i)
    {
        const long k = static_cast<long>(key[i]) - static_cast<long>(scan_grid_min_[i]);
        if (k < 0 || k >= scan_grid_size_[i])
        {
            return -1;
        }
        index = index * scan_grid_size_[i] + k;
    }

    return index;
}

void ScanIntegrator::castRay(const octomap::OcTree &octree, const octomap::point3d &origin, const octomap::point3d &end,
                             bool include_end)
{
    // clip the ray between the scan heights
    octomap::point3d start = origin;
    octomap::point3d stop = end;
    bool clipped_end = false;
    const double dz = end.z() - origin.z();

    if (std::abs(dz) > 1e-9)
    {
        double t_min = (scan_min_z_ - origin.z()) / dz;
        double t_max = (scan_max_z_ - origin.z()) / dz;
        if (t_min > t_max)
        {
            std::swap(t_min, t_max);
        }
        t_min = std::max(t_min, 0.0);
        t_max = std::min(t_max, 1.0);

        if (t_min >= t_max)
        {
            return;
        }
        if (t_min > 0.0)
        {
            start = origin + (end - origin) * t_min;
        }
        if (t_max < 1.0)
        {
            // the ray goes on above or below the band: the last voxel inside it is free
            stop = origin + (end - origin) * t_max;
            include_end = true;
            clipped_end = true;
        }
    }
    else if (origin.z() < scan_min_z_ || origin.z() > scan_max_z_)
    {
        return;
    }

    if (octree.computeRayKeys(start, stop, key_ray_))
    {
        for (const octomap::OcTreeKey &key : key_ray_)
        {
            markFree(key);
        }
    }

    octomap::OcTreeKey stop_key;
    if (include_end && octree.coordToKeyChecked(stop, stop_key))
    {
        // a clipped end lies on the face of the band and may round to the voxel just out
        // of it: the voxel the ray leaves the band through is the one inside
        if (clipped_end)
        {
            const long grid_min_z = scan_grid_min_[2];
            const long grid_max_z = grid_min_z + scan_grid_size_[2] - 1;
            if (static_cast<long>(stop_key[2]) == grid_max_z + 1)
            {
                stop_key[2]--;
            }
            else if (static_cast<long>(stop_key[2]) == grid_min_z - 1)
            {
                stop_key[2]++;
            }
        }
        markFree(stop_key);
    }
}

void ScanIntegrator::markFree(const octomap::OcTreeKey &key)
{
    const long index = scanGridIndex(key);
    if (index < 0)
    {
        return;
    }

    // voxels another sensor is responsible for
    if (key[2] >= scan_keep_min_key_ && key[2] <= scan_keep_max_key_)
    {
        return;
    }

    // skip cells already collected and cells holding a return of this scan
    uint32_t &cell = scan_grid_[index];
    if (cell == 2 * scan_id_ || cell == 2 * scan_id_ + 1)
    {
        return;
    }

    cell = 2 * scan_id_;
    free_cells_.push_back(key);
}

void ScanIntegrator::decay(octomap::OcTree &octree, double stamp)
{
    // going through the history once per second is enough
    if (stamp >= last_decay_stamp_ && stamp - last_decay_stamp_ < 1.0)
    {
        return;
    }
    last_decay_stamp_ = stamp;

    for (auto it = last_hit_stamps_.begin(); it != last_hit_stamps_.end();)
    {
        if (it->second > stamp)
        {
            // the clock jumped back (e.g. simulation reset), restart the count
            it->second = stamp;
        }

        if (stamp - it->second < decay_time_)
        {
            ++it;
            continue;
        }

        // not observed for decay_time: clear it if it is still an obstacle. It is set free
        // rather than deleted, OcTree::deleteNode leaks the children of the emptied nodes.
        octomap::OcTreeNode *node = octree.search(it->first);
        if (node != NULL && octree.isNodeOccupied(node))
        {
            octree.setNodeValue(it->first, octree.getClampingThresMinLog());
        }
        it = last_hit_stamps_.erase(it);
    }
}

// ! SETTERS

void ScanIntegrator::setDecayTime(double decay_time)
{
    decay_time_ = decay_time;
    if (decay_time_ <= 0.0)
    {
        last_hit_stamps_.clear();
    }
}
