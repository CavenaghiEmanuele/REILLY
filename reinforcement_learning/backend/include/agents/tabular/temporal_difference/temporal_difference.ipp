#pragma once

#include "temporal_difference.hpp"

namespace rl {

namespace agents {

TemporalDifference::TemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : Agent(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

TemporalDifference::TemporalDifference(const TemporalDifference &other) : Agent(other) {}

TemporalDifference::~TemporalDifference() {}

void TemporalDifference::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, init_state);
}

}  // namespace agents

}  // namespace rl
