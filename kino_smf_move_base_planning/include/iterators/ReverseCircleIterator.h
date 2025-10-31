#ifndef GRID_MAP_REVERSE_CIRCLE_ITERATOR_H
#define GRID_MAP_REVERSE_CIRCLE_ITERATOR_H

#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/iterators/CircleIterator.hpp>
#include <vector>
#include <algorithm>

namespace grid_map
{

    /**
     * @brief Iterator that traverses cells within a circular region,
     *        starting from the outer boundary and moving inward.
     *
     * Example usage:
     * @code
     * grid_map::Position center(0.0, 0.0);
     * double radius = 1.0;
     * for (grid_map::ReverseCircleIterator it(map, center, radius);
     *      !it.isPastEnd(); ++it)
     * {
     *     grid_map::Index index(*it);
     *     // process cells from outermost to innermost
     * }
     * @endcode
     */
    class ReverseCircleIterator
    {
    public:
        /**
         * @brief Construct an iterator that visits cells from outer to inner radius.
         * @param map     The grid map to iterate over.
         * @param center  The circular region's center position (x, y).
         * @param radius  The radius of the circular region.
         */
        ReverseCircleIterator(const grid_map::GridMap &map,
                              const grid_map::Position &center,
                              double radius);

        /// Returns true if the iterator has reached the end.
        bool isPastEnd() const;

        /// Dereference operator returns the current cell index.
        const grid_map::Index &operator*() const;

        /// Pre-increment operator (advance to next cell).
        ReverseCircleIterator &operator++();

    private:
        const grid_map::GridMap &map_;
        grid_map::Position center_;
        double radius_;

        std::vector<std::pair<double, grid_map::Index>> sorted_cells_; // (distance, index)
        std::size_t current_index_;
    };

} // namespace grid_map

#endif // GRID_MAP_REVERSE_CIRCLE_ITERATOR_H
