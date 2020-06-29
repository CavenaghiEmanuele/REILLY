#pragma once

#include "temporal_difference.hpp"

namespace rl {

namespace agents {

ApproximateTemporalDifference::ApproximateTemporalDifference(size_t actions, float alpha, float epsilon, float gamma,
                                                             float epsilon_decay, size_t tilings,
                                                             std::list<float> tilings_offset, std::list<float> tile_size)
    : ApproximateAgent(actions, alpha, epsilon, gamma, epsilon_decay, tilings, tilings_offset, tile_size) {}

ApproximateTemporalDifference::ApproximateTemporalDifference(const ApproximateTemporalDifference &other)
    : ApproximateAgent(other) {}

ApproximateTemporalDifference::~ApproximateTemporalDifference() {}

void ApproximateTemporalDifference::reset(State init_state) {
    state = init_state;
    action = select_action(estimator, init_state);
}

}  // namespace agents

}  // namespace rl