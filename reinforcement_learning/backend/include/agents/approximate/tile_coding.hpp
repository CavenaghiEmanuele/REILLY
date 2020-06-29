#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <vector>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>

namespace rl {

namespace agents {

using State = xt::xtensor<float, 1>;
using Coordinates = xt::xtensor<size_t, 1>;

size_t Tile(State &start_point, State &end_point, size_t &action);

class Tiling {
   private:
    State tile_size;
    State start_point;
    std::map<size_t, float> weights;

   public:
    Tiling(State tile_size, State start_point);
    Tiling(const Tiling &other);
    Tiling &operator=(const Tiling &other);
    ~Tiling();

    size_t operator()(State &features, size_t action);
    float operator()(size_t coordinate);

    void update(size_t coordinate, float weight);
};

class TileCoding {
   private:
    float alpha;
    std::vector<Tiling> tilings;

   public:
    TileCoding(float alpha, size_t tilings, State tilings_offset, State tile_size);
    TileCoding(const TileCoding &other);
    TileCoding &operator=(const TileCoding &other);
    ~TileCoding();

    float operator()(State &state, size_t action);
    void update(State &state, size_t action, float reward);
};

}  // namespace agents

}  // namespace rl
