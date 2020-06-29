#pragma once

#include "agent.hpp"

namespace rl {

namespace agents {

Agent::Agent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : actions(actions), alpha(alpha), epsilon(epsilon), gamma(gamma), epsilon_decay(epsilon_decay) {
    generator.seed(time(NULL));
    xt::random::seed(time(NULL));
}

Agent::Agent(const Agent &other)
    : actions(other.actions),
      alpha(other.alpha),
      epsilon(other.epsilon),
      gamma(other.gamma),
      epsilon_decay(other.epsilon_decay),
      action(other.action) {}

Agent::~Agent() {}

size_t Agent::get_action() { return action; }

}  // namespace agents

}  // namespace rl
