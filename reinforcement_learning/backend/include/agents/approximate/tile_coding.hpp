#pragma once

#include "../agent.ipp"

namespace rl {

namespace agents {

size_t Tile(Vector &start_point, Vector &end_point, size_t &action);

class Tiling {
   private:
    Vector tile_size;
    Vector start_point;
    std::map<size_t, float> weights;

   public:
    Tiling(Vector tile_size, Vector start_point);
    Tiling(const Tiling &other);
    Tiling &operator=(const Tiling &other);
    ~Tiling();

    size_t operator()(Vector &features, size_t action);
    float operator()(size_t coordinate);

    void update(size_t coordinate, float weight);
};

class TileCoding {
   private:
    size_t actions;
    float alpha;
    size_t features;
    std::vector<Tiling> tilings;

    // Pre-allocated temporary memory
    Vector _features;
    xt::xtensor<size_t, 1> _coordinates;

   public:
    TileCoding(size_t actions, float alpha, size_t features, size_t tilings, Vector tilings_offset, Vector tile_size);
    TileCoding(const TileCoding &other);
    TileCoding &operator=(const TileCoding &other);
    ~TileCoding();

    float operator()(Vector &state, size_t action);
    void update(Vector &state, size_t action, float reward);
};

}  // namespace agents

}  // namespace rl
