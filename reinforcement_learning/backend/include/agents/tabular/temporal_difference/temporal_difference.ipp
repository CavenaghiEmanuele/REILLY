#pragma once

#include "temporal_difference.hpp"

namespace rl {

namespace agents {

TemporalDifference::TemporalDifference(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : Agent(states, actions, epsilon, gamma, epsilon_decay), alpha(alpha) {}

TemporalDifference::TemporalDifference(const TemporalDifference &other) : Agent(other), alpha(other.alpha) {}

TemporalDifference::~TemporalDifference() {}

void TemporalDifference::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, state);
}

std::string TemporalDifference::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
