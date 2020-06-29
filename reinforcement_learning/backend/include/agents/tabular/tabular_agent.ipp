#pragma once

#include "tabular_agent.hpp"

namespace rl {

namespace agents {

TabularAgent::TabularAgent(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : Agent(actions, alpha, epsilon, gamma, epsilon_decay), states(states) {
    Q = xt::zeros<float>({states, actions});
    pi = xt::ones<float>({states, actions}) / actions;
}

TabularAgent::TabularAgent(const TabularAgent &other)
    : Agent(other), states(other.states), Q(other.Q), pi(other.pi), state(other.state) {}

TabularAgent::~TabularAgent() {}

inline size_t TabularAgent::argmaxQs(const ActionValue &Q, size_t state) {
    xt::xtensor<float, 1> row = xt::row(Q, state);
    return xt::random::choice(xt::ravel_indices(xt::argwhere(xt::equal(row, xt::amax(row))), row.shape()), 1)(0);
}

inline size_t TabularAgent::select_action(const Policy &pi, size_t state) {
    xt::xtensor<float, 1> weights = xt::row(pi, state);
    std::discrete_distribution<size_t> distribution(weights.cbegin(), weights.cend());
    return distribution(generator);
}

inline void TabularAgent::policy_update(const ActionValue &Q, Policy &pi, size_t state) {
    // Select greedy action, ties broken arbitrarily
    size_t a_star = argmaxQs(Q, state);
    // Update policy
    for (size_t a = 0; a < pi.shape(1); a++) {
        if (a == a_star)
            pi(state, a) = 1 - epsilon + epsilon / actions;
        else
            pi(state, a) = epsilon / actions;
    }
}

std::string TabularAgent::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
