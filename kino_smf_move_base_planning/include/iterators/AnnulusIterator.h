#ifndef GRID_MAP_ANNULUS_ITERATOR_H
#define GRID_MAP_ANNULUS_ITERATOR_H

#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/iterators/CircleIterator.hpp>

namespace grid_map
{

    /**
     * @brief Iterator for traversing cells in an annular (ring-shaped) region
     *        between an inner and an outer radius.
     *
     * Example usage:
     * @code
     * grid_map::Position center(1.0, 2.0);
     * double inner = 0.2, outer = 0.5;
     * for (grid_map::AnnulusIterator it(map, center, inner, outer);
     *      !it.isPastEnd(); ++it)
     * {
     *     grid_map::Index index(*it);
     *     // ... your code ...
     * }
     * @endcode
     */
    class AnnulusIterator
    {
    public:
        /**
         * @brief Construct an iterator over the annular region.
         * @param map           The grid map to iterate over.
         * @param center        Center position (x, y).
         * @param inner_radius  Inner radius to exclude.
         * @param outer_radius  Outer radius to include.
         */
        AnnulusIterator(const grid_map::GridMap &map,
                        const grid_map::Position &center,
                        double inner_radius,
                        double outer_radius);

        /// Returns true if the iterator has reached the end.
        bool isPastEnd() const;

        /// Dereference operator returns the current cell index.
        const grid_map::Index &operator*() const;

        /// Pre-increment operator.
        AnnulusIterator &operator++();

    private:
        /// Skips cells that are inside the inner radius.
        void skipInner();

        grid_map::CircleIterator iterator_;
        const grid_map::GridMap &map_;
        grid_map::Position center_;
        double inner_radius_;
        double outer_radius_;
    };

} // namespace grid_map

#endif // GRID_MAP_ANNULUS_ITERATOR_H
