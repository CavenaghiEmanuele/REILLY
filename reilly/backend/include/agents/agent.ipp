#pragma once

#include "agent.hpp"

namespace reilly {

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

inline size_t Agent::argmaxQs(Vector &w) {
    return xt::random::choice(xt::ravel_indices(xt::argwhere(xt::equal(w, xt::amax(w))), w.shape()), 1)(0);
}

inline Vector Agent::e_greedy_policy(Vector &weights) {
    Vector out = xt::empty<float>({actions});
    // Select greedy action, ties broken arbitrarily
    size_t a_star = argmaxQs(weights);
    // Update policy
    for (size_t a = 0; a < actions; a++) {
        if (a == a_star)
            out(a) = 1 - epsilon + epsilon / actions;
        else
            out(a) = epsilon / actions;
    }
    return out;
}

inline size_t Agent::select_action(Vector &weights) {
    std::discrete_distribution<size_t> distribution(weights.cbegin(), weights.cend());
    return distribution(generator);
}

size_t Agent::get_action() { return action; }

}  // namespace agents

}  // namespace reilly
