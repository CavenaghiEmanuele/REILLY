#pragma once

#include "agent.hpp"

namespace rl {

namespace agents {

Agent::Agent(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : states(states), actions(actions), epsilon(epsilon), gamma(gamma), epsilon_decay(epsilon_decay) {
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

size_t Agent::get_action() {
    xt::xtensor<float, 1> weights = xt::row(pi, state);
    std::discrete_distribution<size_t> distribution(weights.cbegin(), weights.cend());
    action = distribution(generator);
    return action;
}

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
