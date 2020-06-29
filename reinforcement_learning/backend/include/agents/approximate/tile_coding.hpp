#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>

namespace rl {

namespace agents {

using State = xt::xtensor<float, 1>;
using Coordinates = xt::xtensor<size_t, 1>;

class Tile {
   private:
    size_t id;

   public:
    Tile(State &start_point, State &end_point, size_t &action);
    Tile(const Tile &other);
    Tile &operator=(const Tile &other);
    ~Tile();

    bool operator==(const Tile &other) const;
    bool operator!=(const Tile &other) const;
};

class Tiling {
   private:
    State tile_size;
    State start_point;
    std::vector<Tile> tiles;
    std::unordered_map<size_t, float> weights;

   public:
    Tiling(State tile_size, State start_point);
    Tiling(const Tiling &other);
    Tiling &operator=(const Tiling &other);
    ~Tiling();

    size_t operator()(State &features, size_t action);
    float operator()(size_t coordinate);

    void update(size_t coordinate, float weight);
    void reset();
};

class TileCoding {
   private:
    float alpha;
    std::vector<Tiling> tilings;

   public:
    TileCoding(float alpha, size_t tilings, State tiling_offset, State tile_size);
    TileCoding(const TileCoding &other);
    TileCoding &operator=(const TileCoding &other);
    ~TileCoding();

    float operator()(State &state, size_t action);
    void update(State &state, size_t action, float reward);
    void reset();
};

}  // namespace agents

}  // namespace rl
