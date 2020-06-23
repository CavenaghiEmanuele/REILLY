#pragma once

#include "agent.hpp"

namespace rl {

namespace agents {

Agent::Agent(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : states(states), actions(actions), epsilon(epsilon), gamma(gamma), epsilon_decay(epsilon_decay) {
    generator.seed(time(NULL));
    xt::random::seed(time(NULL));
    Q = xt::zeros<float>({states, actions});
    pi = xt::ones<float>({states, actions}) / actions;
}

Agent::Agent(const Agent &other)
    : states(other.states),
      actions(other.actions),
      Q(other.Q),
      pi(other.pi),
      epsilon(other.epsilon),
      gamma(other.gamma),
      epsilon_decay(other.epsilon_decay),
      state(other.state),
      action(other.action) {}

Agent::~Agent() {}

inline size_t Agent::argmaxQs(const ActionValue &Qs, size_t s) {
    xt::xtensor<float, 1> row = xt::row(Qs, s);
    return xt::random::choice(xt::ravel_indices(xt::argwhere(xt::equal(row, xt::amax(row))), row.shape()), 1)(0);
}

inline size_t Agent::select_action(size_t s) {
    xt::xtensor<float, 1> weights = xt::row(pi, s);
    std::discrete_distribution<size_t> distribution(weights.cbegin(), weights.cend());
    return distribution(generator);
}

inline void Agent::policy_update(size_t s) {
    // Select greedy action, ties broken arbitrarily
    size_t a_star = argmaxQs(Q, s);
    // Update policy
    for (size_t a = 0; a < pi.shape(1); a++) {
        if (a == a_star)
            pi(s, a) = 1 - epsilon + epsilon / actions;
        else
            pi(s, a) = epsilon / actions;
    }
}

size_t Agent::get_action() { return action; }

std::string Agent::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
