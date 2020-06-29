#pragma once

#include "tile_coding.hpp"

namespace rl {

namespace agents {

Tile::Tile(State &start_point, State &end_point, size_t &action) {
    std::stringstream hash;
    hash << start_point << ",";
    hash << end_point << ",";
    hash << action;
    id = std::hash<std::string>()(hash.str());
}

Tile::Tile(const Tile &other) : id(other.id) {}

Tile &Tile::operator=(const Tile &other) {
    if (this != &other) {
        Tile tmp(other);
        std::swap(tmp.id, id);
    }
    return *this;
}

Tile::~Tile() {}

bool Tile::operator==(const Tile &other) const { return id == other.id; }

bool Tile::operator!=(const Tile &other) const { return !(this == &other); }

Tiling::Tiling(State tile_size, State start_point) : tile_size(tile_size), start_point(start_point) {}

Tiling::Tiling(const Tiling &other)
    : tile_size(other.tile_size), start_point(other.start_point), tiles(other.tiles), weights(other.weights) {}

Tiling &Tiling::operator=(const Tiling &other) {
    if (this != &other) {
        Tiling tmp(other);
        std::swap(tmp.tile_size, tile_size);
        std::swap(tmp.start_point, start_point);
        std::swap(tmp.tiles, tiles);
        std::swap(tmp.weights, weights);
    }
    return *this;
}

Tiling::~Tiling() {}

size_t Tiling::operator()(State &features, size_t action) {
    auto copy_sign = xt::vectorize(std::copysign<float, float>);
    State lower_bound = copy_sign(xt::floor(xt::abs(features - start_point) / tile_size), features);
    State end_point = lower_bound + tile_size;
    Tile tile(lower_bound, end_point, action);
    std::vector<Tile>::iterator it = std::find(tiles.begin(), tiles.end(), tile);
    if (it == tiles.end()) {
        tiles.push_back(tile);
        return tiles.size();
    }
    return std::distance(tiles.begin(), it);
}

float Tiling::operator()(size_t coordinate) {
    auto it = weights.find(coordinate);
    if (it == weights.end()) return 0;
    return it->second;
}

void Tiling::update(size_t coordinate, float value) { weights[coordinate] = value; }

void Tiling::reset() {
    tiles.clear();
    weights.clear();
}

TileCoding::TileCoding(float alpha, size_t tilings, State tile_size, State tiling_offset) : alpha(alpha / tilings) {
    for (size_t i = 0; i < tilings; i++) {
        this->tilings.push_back(Tiling(tile_size, (-i) * tiling_offset));
    }
}

TileCoding::TileCoding(const TileCoding &other) : tilings(other.tilings) {}

TileCoding &TileCoding::operator=(const TileCoding &other) {
    if (this != &other) {
        TileCoding tmp(other);
        std::swap(tmp.tilings, tilings);
    }
    return *this;
}

TileCoding::~TileCoding() {}

float TileCoding::operator()(State &state, size_t action) {
    State features = xt::empty<float>({state.shape(0)});
    for (size_t i = 0; i < state.shape(0); i++) {
        size_t coordinate = tilings[i](state, action);
        features(i) = tilings[i](coordinate);
    }
    // Linear Function Approximation
    return xt::sum(features)();
}

void TileCoding::update(State &state, size_t action, float reward) {
    Coordinates coordinates = xt::empty<size_t>({state.shape(0)});
    State features = xt::empty<float>({state.shape(0)});
    for (size_t i = 0; i < state.shape(0); i++) {
        coordinates(i) = tilings[i](state, action);
        features(i) = tilings[i](coordinates(i));
    }
    // Linear Function Approximation
    float delta = reward - xt::sum(features)();
    for (size_t i = 0; i < state.shape(0); i++) {
        tilings[i].update(coordinates(i), features(i) + alpha * delta);
    }
}

void TileCoding::reset() {
    for (size_t i = 0; i < tilings.size(); i++) {
        tilings[i].reset();
    }
}

}  // namespace agents

}  // namespace rl