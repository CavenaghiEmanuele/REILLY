#pragma once

#include "../agent.ipp"

namespace rl {

namespace agents {

using Coordinates = xt::xtensor<size_t, 1>;

size_t Tile(const Vector &start_point, const Vector &end_point, size_t action);

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

    size_t operator()(const Vector &state, size_t action) const;
    float operator()(size_t coordinate) const;

    void update(size_t coordinate, float weight);
};

class TileCoding {
   private:
    size_t actions;
    float alpha;
    size_t features;
    std::vector<Tiling> tilings;

   public:
    TileCoding(size_t actions, float alpha, py::kwargs kwargs);
    TileCoding(const TileCoding &other);
    TileCoding &operator=(const TileCoding &other);
    ~TileCoding();

    Vector operator()(const Vector &state) const;
    float operator()(const Vector &state, size_t action) const;
    void update(const Vector &state, size_t action, float reward);

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl
