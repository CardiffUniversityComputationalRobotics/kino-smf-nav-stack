/*! \file state_validity_checker_octomap_fcl_R2.cpp
 * \brief State validity checker.
 *
 * \date March 5, 2015
 * \author Juan David Hernandez Vega, juandhv@rice.edu
 *
 * \details Check is a given configuration R2 is collision-free.
 *  The worspace is represented by an Octomap and collision check is done with FCL.
 *
 * Based on Juan D. Hernandez Vega's PhD thesis, University of Girona
 * http://hdl.handle.net/10803/457592, http://www.tdx.cat/handle/10803/457592
 */

#include <local_state_validity_checker_grid_map_SE2.h>

LocalGridMapStateValidityCheckerSE2::LocalGridMapStateValidityCheckerSE2(const ob::SpaceInformationPtr &si,
                                                                         const bool opport_collision_check,
                                                                         std::vector<double> planning_bounds_x,
                                                                         std::vector<double> planning_bounds_y, grid_map::GridMap grid_map, const double robot_radius, const bool local_use_social_heatmap, std::shared_ptr<octomap::OcTree> octree)
    : ob::StateValidityChecker(si), robot_base_radius_(0.4)
{

    opport_collision_check_ = opport_collision_check;
    planning_bounds_x_ = planning_bounds_x;
    planning_bounds_y_ = planning_bounds_y;
    robot_base_radius_ = robot_radius;
    local_use_social_heatmap_ = local_use_social_heatmap;

    robot_base_height_ = 1.5;

    grid_map_ = grid_map;

    try
    {
        full_grid_map_ = grid_map_.get("full");
        comfort_grid_map_ = grid_map_.get("comfort");
        social_heatmap_grid_map_ = grid_map_.get("social_heatmap");
        obstacles_grid_map_ = grid_map_.get("obstacles");

        // OCTOMAP PROCESS
        octree_ = octree;
        tree_ = new fcl::OcTreef(octree_);
        tree_obj_ = new fcl::CollisionObjectf((std::shared_ptr<fcl::CollisionGeometryf>(tree_)));

        robot_collision_solid_.reset(new fcl::Cylinderf(robot_base_radius_, robot_base_height_));

        octree_res_ = octree->getResolution();
        octree_->getMetricMin(octree_min_x_, octree_min_y_, octree_min_z_);
        octree_->getMetricMax(octree_max_x_, octree_max_y_, octree_max_z_);
    }
    catch (...)
    {
    }
}

bool LocalGridMapStateValidityCheckerSE2::isValid(const ob::State *state) const
{

    const ob::SE2StateSpace::StateType *state_se2 = state->as<ob::SE2StateSpace::StateType>();

    if (opport_collision_check_ &&
        (state_se2->getX() < octree_min_x_ || state_se2->getY() < octree_min_y_ ||
         state_se2->getX() > octree_max_x_ || state_se2->getY() > octree_max_y_))
    {
        return true;
    }

    if (state_se2->getX() < planning_bounds_x_[0] || state_se2->getY() < planning_bounds_y_[0] ||
        state_se2->getX() > planning_bounds_x_[1] || state_se2->getY() > planning_bounds_y_[1])
    {
        return false;
    }

    // FCL
    fcl::Transform3f robot_tf;
    robot_tf.setIdentity();
    robot_tf.translate(fcl::Vector3f(state_se2->getX(), state_se2->getY(), robot_base_height_ / 2.0));

    fcl::CollisionObjectf vehicle_co(robot_collision_solid_, robot_tf);

    fcl::CollisionRequestf collision_request;
    fcl::CollisionResultf collision_result;

    fcl::collide(tree_obj_, &vehicle_co, collision_request, collision_result);

    // std::cout << "Collision (FCL): " << collision_result.isCollision() << std::endl;

    if (collision_result.isCollision())
    {
        // ompl::tools::Profiler::End("collision");
        return false;
    }

    return true;
}

double LocalGridMapStateValidityCheckerSE2::checkExtendedSocialComfort(const ob::State *state,
                                                                       const ob::SpaceInformationPtr space) const
{

    double state_risk = 0.0;
    grid_map::Index index;

    const ob::SE2StateSpace::StateType *state_se2 = state->as<ob::SE2StateSpace::StateType>();
    grid_map::Position query(state_se2->getX(), state_se2->getY());

    if (grid_map_.getIndex(query, index))
    {
        state_risk = comfort_grid_map_(index(0), index(1));

        if (state_risk < 1 || isnan(state_risk))
        {
            state_risk = 1;
        }

        if (local_use_social_heatmap_)
        {
            double social_heatmap_risk = social_heatmap_grid_map_(index(0), index(1));
            if (!isnan(social_heatmap_risk) && !social_heatmap_risk <= 1)
            {
                state_risk += social_heatmap_risk / 100;
            }
        }

        bool is_risk_zone = false;

        for (grid_map::CircleIterator iterator(grid_map_, query, robot_base_radius_ + 0.3);
             !iterator.isPastEnd(); ++iterator)
        {
            const grid_map::Index index(*iterator);

            if (obstacles_grid_map_(index(0), index(1)) > 20)
            {
                is_risk_zone = true;
                break;
            }
        }

        if (is_risk_zone)
        {
            state_risk += 10;
        }
        else
        {
            for (grid_map::CircleIterator iterator(grid_map_, query, robot_base_radius_ + 0.15);
                 !iterator.isPastEnd(); ++iterator)
            {
                const grid_map::Index index(*iterator);

                if (obstacles_grid_map_(index(0), index(1)) > 20)
                {
                    is_risk_zone = true;
                    break;
                }
            }

            if (is_risk_zone)
            {
                state_risk += 5;
            }
        }
    }

    return state_risk;
}

bool LocalGridMapStateValidityCheckerSE2::isValidPoint(const ob::State *state) const
{
    OcTreeNode *result;
    point3d query;
    double node_occupancy;

    // extract the component of the state and cast it to what we expect
    const ob::SE2StateSpace::StateType *state_se2 = state->as<ob::SE2StateSpace::StateType>();

    query.x() = state_se2->getX();
    query.y() = state_se2->getY();
    query.z() = 0.0;

    result = octree_->search(query);

    if (result == NULL)
    {
        return false;
    }
    else
    {
        node_occupancy = result->getOccupancy();
        if (node_occupancy <= 0.2)
            return true;
    }
    return false;
}

LocalGridMapStateValidityCheckerSE2::~LocalGridMapStateValidityCheckerSE2()
{
}
