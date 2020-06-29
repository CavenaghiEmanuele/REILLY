#pragma once

#include "tile_coding.hpp"

namespace rl {

namespace agents {

size_t Tile(Vector &start_point, Vector &end_point, size_t &action) {
    std::stringstream hash;
    hash << start_point << ',';
    hash << end_point << ',';
    hash << action;
    return std::hash<std::string>()(hash.str());
}

Tiling::Tiling(Vector tile_size, Vector start_point) : tile_size(tile_size), start_point(start_point) {}

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

size_t Tiling::operator()(Vector &features, size_t action) {
    auto copy_sign = xt::vectorize(std::copysign<float, float>);
    Vector lower_bound = copy_sign(xt::floor(xt::abs(features - start_point) / tile_size), features);
    Vector end_point = lower_bound + tile_size;
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

TileCoding::TileCoding(size_t actions, float alpha, size_t features, size_t tilings, Vector tile_size, Vector tilings_offset)
    : actions(actions), alpha(alpha / tilings), features(features) {
    _features = xt::empty<float>({features});
    _coordinates = xt::empty<size_t>({features});
    for (size_t i = 0; i < tilings; i++) {
        this->tilings.push_back(Tiling(tile_size, (-i) * tilings_offset));
    }
}

TileCoding::TileCoding(const TileCoding &other) : tilings(other.tilings) {}

TileCoding &TileCoding::operator=(const TileCoding &other) {
    if (this != &other) {
        TileCoding tmp(other);
        std::swap(tmp.actions, actions);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.features, features);
        std::swap(tmp.tilings, tilings);
        std::swap(tmp._features, _features);
        std::swap(tmp._coordinates, _coordinates);
    }
    return *this;
}

TileCoding::~TileCoding() {}

float TileCoding::operator()(Vector &state, size_t action) {
    size_t _coordinate = 0;
    for (size_t i = 0; i < features; i++) {
        _coordinate = tilings[i](state, action);
        _features(i) = tilings[i](_coordinate);
    }
    // Linear Function Approximation
    return xt::sum(_features)();
}

void TileCoding::update(Vector &state, size_t action, float target) {
    for (size_t i = 0; i < features; i++) {
        _coordinates(i) = tilings[i](state, action);
        _features(i) = tilings[i](_coordinates(i));
    }
    // Linear Function Approximation
    float delta = target - xt::sum(_features)();
    for (size_t i = 0; i < features; i++) {
        tilings[i].update(_coordinates(i), _features(i) + alpha * delta);
    }
}

}  // namespace agents

}  // namespace rl