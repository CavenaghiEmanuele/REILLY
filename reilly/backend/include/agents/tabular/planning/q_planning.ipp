#pragma once

#include "q_planning.hpp"

namespace reilly {

namespace agents {

QPlanning::QPlanning(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float epsilon_decay)
    : TabularAgent(states, actions, alpha, epsilon, gamma, epsilon_decay), n_plan(n_plan), model(states, actions) {}

QPlanning::QPlanning(const QPlanning &other) : TabularAgent(other), n_plan(other.n_plan), model(other.model) {}

QPlanning::~QPlanning() {}

void QPlanning::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, init_state);
}

std::string QPlanning::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", n_plan=" << n_plan << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace reilly