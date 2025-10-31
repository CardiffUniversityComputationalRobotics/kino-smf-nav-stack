#include <iterators/AnnulusIterator.h>

namespace grid_map
{

    AnnulusIterator::AnnulusIterator(const grid_map::GridMap &map,
                                     const grid_map::Position &center,
                                     double inner_radius,
                                     double outer_radius)
        : iterator_(map, center, outer_radius),
          map_(map),
          center_(center),
          inner_radius_(inner_radius),
          outer_radius_(outer_radius)
    {
        skipInner();
    }

    bool AnnulusIterator::isPastEnd() const
    {
        return iterator_.isPastEnd();
    }

    const grid_map::Index &AnnulusIterator::operator*() const
    {
        return *iterator_;
    }

    AnnulusIterator &AnnulusIterator::operator++()
    {
        ++iterator_;
        skipInner();
        return *this;
    }

    void AnnulusIterator::skipInner()
    {
        while (!iterator_.isPastEnd())
        {
            grid_map::Position pos;
            map_.getPosition(*iterator_, pos);
            double dist = (pos - center_).norm();
            if (dist >= inner_radius_)
                break;
            ++iterator_;
        }
    }

} // namespace grid_map
