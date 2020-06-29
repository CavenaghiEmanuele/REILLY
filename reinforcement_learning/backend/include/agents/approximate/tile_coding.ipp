#pragma once

#include "tile_coding.hpp"

namespace rl {

namespace agents {

size_t Tile(State &start_point, State &end_point, size_t &action) {
    std::stringstream hash;
    hash << start_point << ',';
    hash << end_point << ',';
    hash << action;
    return std::hash<std::string>()(hash.str());
}

Tiling::Tiling(State tile_size, State start_point) : tile_size(tile_size), start_point(start_point) {}

Tiling::Tiling(const Tiling &other)
    : tile_size(other.tile_size), start_point(other.start_point), weights(other.weights) {}

Tiling &Tiling::operator=(const Tiling &other) {
    if (this != &other) {
        Tiling tmp(other);
        std::swap(tmp.tile_size, tile_size);
        std::swap(tmp.start_point, start_point);
        std::swap(tmp.weights, weights);
    }
    return *this;
}

Tiling::~Tiling() {}

size_t Tiling::operator()(State &features, size_t action) {
    auto copy_sign = xt::vectorize(std::copysign<float, float>);
    State lower_bound = copy_sign(xt::floor(xt::abs(features - start_point) / tile_size), features);
    State end_point = lower_bound + tile_size;
    return Tile(lower_bound, end_point, action);
}

float Tiling::operator()(size_t coordinate) {
    auto it = weights.find(coordinate);
    if (it == weights.end()) return 0;
    return it->second;
}

void Tiling::update(size_t coordinate, float value) {
    weights[coordinate] = value;
}

TileCoding::TileCoding(float alpha, size_t tilings, State tile_size, State tilings_offset) : alpha(alpha / tilings) {
    for (size_t i = 0; i < tilings; i++) {
        this->tilings.push_back(Tiling(tile_size, (-i) * tilings_offset));
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

void TileCoding::update(State &state, size_t action, float target) {
    Coordinates coordinates = xt::empty<size_t>({state.shape(0)});
    State features = xt::empty<float>({state.shape(0)});
    for (size_t i = 0; i < state.shape(0); i++) {
        coordinates(i) = tilings[i](state, action);
        features(i) = tilings[i](coordinates(i));
    }
    // Linear Function Approximation
    float delta = target - xt::sum(features)();
    for (size_t i = 0; i < state.shape(0); i++) {
        tilings[i].update(coordinates(i), features(i) + alpha * delta);
    }
}

}  // namespace agents

}  // namespace rl