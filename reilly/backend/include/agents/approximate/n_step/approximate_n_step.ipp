#pragma once

#include "approximate_n_step.hpp"

namespace rl {

namespace agents {

ApproximateNStep::ApproximateNStep(size_t actions, float alpha, float epsilon, float gamma, size_t n_step,
                                   float epsilon_decay, py::kwargs kwargs)
    : ApproximateAgent(actions, alpha, epsilon, gamma, epsilon_decay, kwargs), n_step(n_step) {}

ApproximateNStep::ApproximateNStep(const ApproximateNStep &other)
    : ApproximateAgent(other), n_step(other.n_step), T(other.T), trajectory(other.trajectory) {}

ApproximateNStep::~ApproximateNStep() {}

void ApproximateNStep::reset(Vector init_state) {
    state = init_state;
    action = select_action(estimator, init_state);

    T = (size_t)-1;

    trajectory.clear();
    trajectory.push_back({state, action, 0});
}

std::string ApproximateNStep::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha= " << alpha << "epsilon=" << epsilon << ", gamma=" << gamma;
    out << ", n_step=" << n_step << ", epsilon_decay=" << epsilon_decay << ", estimator=" << estimator.__repr__();
    return out.str();
}

}  // namespace agents

}  // namespace rl