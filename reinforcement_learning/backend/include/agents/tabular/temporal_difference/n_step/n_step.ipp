#pragma once

#include "n_step.hpp"

namespace rl {

namespace agents {

NStep::NStep(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay), n_step(n_step) {}

NStep::NStep(const NStep &other) : TemporalDifference(other), n_step(other.n_step), trajectory(other.trajectory) {}

NStep::~NStep() {}

std::string NStep::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", n_step=" << n_step << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
