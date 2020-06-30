#pragma once

#include "double_temporal_difference.hpp"

namespace rl {

namespace agents {

DoubleTemporalDifference::DoubleTemporalDifference(size_t states, size_t actions, float alpha, float epsilon,
                                                   float gamma, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {
    Q2 = xt::zeros<float>({states, actions});
    pi2 = xt::ones<float>({states, actions}) / actions;
}
DoubleTemporalDifference::DoubleTemporalDifference(const DoubleTemporalDifference &other)
    : TemporalDifference(other), Q2(other.Q2), pi2(other.pi2) {}

DoubleTemporalDifference::~DoubleTemporalDifference() {}

void DoubleTemporalDifference::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi + pi2, state);
}

}  // namespace agents

}  // namespace rl
