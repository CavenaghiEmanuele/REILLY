#pragma once

#include "expected_sarsa.hpp"

namespace rl {

namespace agents {

SemiGradientExpectedSarsa::SemiGradientExpectedSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                                     size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size)
    : ApproximateTemporalDifference(actions, alpha, epsilon, gamma, epsilon_decay, tilings, tilings_offset, tile_size) {}

SemiGradientExpectedSarsa::SemiGradientExpectedSarsa(const SemiGradientExpectedSarsa &other) : ApproximateTemporalDifference(other) {}

SemiGradientExpectedSarsa &SemiGradientExpectedSarsa::operator=(const SemiGradientExpectedSarsa &other) {
    if (this != &other) {
        SemiGradientExpectedSarsa tmp(other);
        std::swap(tmp.action, action);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.estimator, estimator);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
    }
    return *this;
}

SemiGradientExpectedSarsa::~SemiGradientExpectedSarsa() {}

void SemiGradientExpectedSarsa::update(Vector next_state, float reward, bool done, py::kwargs kwargs) {
    size_t next_action = select_action(estimator, next_state);
    
    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        float G = reward + gamma * estimator(next_state, next_action);
        estimator.update(state, action, G);
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl