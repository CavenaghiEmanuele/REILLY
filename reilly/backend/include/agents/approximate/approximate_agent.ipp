#pragma once

#include "approximate_agent.hpp"

namespace reilly {

namespace agents {

ApproximateAgent::ApproximateAgent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, py::kwargs kwargs)
    : Agent(actions, alpha, epsilon, gamma, epsilon_decay),
      estimator(actions, alpha, kwargs) {}

ApproximateAgent::ApproximateAgent(const ApproximateAgent &other) : Agent(other), estimator(other.estimator) {}

ApproximateAgent::~ApproximateAgent() {}

inline size_t ApproximateAgent::select_action(TileCoding &estimator, Vector &state) {
    Vector weights = estimator(state);
    weights = e_greedy_policy(weights);
    return Agent::select_action(weights);
}

void ApproximateAgent::reset(size_t init_state) {
    Vector vector_state = {(float)init_state};
    reset(vector_state);
}

void ApproximateAgent::reset(std::vector<float> init_state) { reset(to_xtensor(init_state)); }

void ApproximateAgent::reset(py::array init_state) {
    auto np = init_state.unchecked<float, 1>();
    Vector vector_state = xt::zeros<float>({np.size()});
    for (long int i = 0; i < np.size(); i++) {
        vector_state[i] = np(i);
    }
    reset(vector_state);
}

void ApproximateAgent::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    Vector vector_state = {(float)next_state};
    update(vector_state, reward, done, kwargs);
}

void ApproximateAgent::update(std::vector<float> next_state, float reward, bool done, py::kwargs kwargs) {
    update(to_xtensor(next_state), reward, done, kwargs);
}

void ApproximateAgent::update(py::array next_state, float reward, bool done, py::kwargs kwargs) {
    auto np = next_state.unchecked<float, 1>();
    Vector vector_state = xt::zeros<float>({np.size()});
    for (long int i = 0; i < np.size(); i++) {
        vector_state[i] = np(i);
    }
    update(vector_state, reward, done, kwargs);
}

std::string ApproximateAgent::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha= " << alpha << ", epsilon=" << epsilon << ", gamma=" << gamma;
    out << ", epsilon_decay=" << epsilon_decay << ", estimator=" << estimator.__repr__() << ")";
    return out.str();
}

}  // namespace agents

}  // namespace reilly
