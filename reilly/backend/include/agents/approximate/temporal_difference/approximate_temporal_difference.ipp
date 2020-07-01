#pragma once

#include "approximate_temporal_difference.hpp"

namespace reilly {

namespace agents {

ApproximateTemporalDifference::ApproximateTemporalDifference(size_t actions, float alpha, float epsilon, float gamma,
                                                             float epsilon_decay, py::kwargs kwargs)
    : ApproximateAgent(actions, alpha, epsilon, gamma, epsilon_decay, kwargs) {}

ApproximateTemporalDifference::ApproximateTemporalDifference(const ApproximateTemporalDifference &other)
    : ApproximateAgent(other) {}

ApproximateTemporalDifference::~ApproximateTemporalDifference() {}

void ApproximateTemporalDifference::reset(Vector init_state) {
    state = init_state;
    action = select_action(estimator, init_state);
}

}  // namespace agents

}  // namespace reilly