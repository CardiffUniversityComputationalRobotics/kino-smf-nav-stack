#include <iterators/ReverseCircleIterator.h>

namespace grid_map
{

    ReverseCircleIterator::ReverseCircleIterator(const grid_map::GridMap &map,
                                                 const grid_map::Position &center,
                                                 double radius)
        : map_(map), center_(center), radius_(radius), current_index_(0)
    {
        // Collect all indices within the circle
        for (grid_map::CircleIterator it(map_, center_, radius_); !it.isPastEnd(); ++it)
        {
            grid_map::Position pos;
            map_.getPosition(*it, pos);
            double dist = (pos - center_).norm();
            sorted_cells_.emplace_back(dist, *it);
        }

        // Sort by distance descending (outer to inner)
        std::sort(sorted_cells_.begin(), sorted_cells_.end(),
                  [](const auto &a, const auto &b)
                  { return a.first > b.first; });
    }

    bool ReverseCircleIterator::isPastEnd() const
    {
        return current_index_ >= sorted_cells_.size();
    }

    const grid_map::Index &ReverseCircleIterator::operator*() const
    {
        return sorted_cells_[current_index_].second;
    }

    ReverseCircleIterator &ReverseCircleIterator::operator++()
    {
        ++current_index_;
        return *this;
    }

} // namespace grid_map
