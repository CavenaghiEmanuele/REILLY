#pragma once

#include "approximate_agent.hpp"

namespace rl {

namespace agents {

ApproximateAgent::ApproximateAgent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, size_t features,
                                   size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size)
    : Agent(actions, alpha, epsilon, gamma, epsilon_decay),
      estimator(actions, alpha, features, tilings, to_xtensor(tilings_offset), to_xtensor(tile_size)) {}

ApproximateAgent::ApproximateAgent(const ApproximateAgent &other) : Agent(other), estimator(other.estimator) {}

ApproximateAgent::~ApproximateAgent() {}

inline size_t ApproximateAgent::select_action(TileCoding &estimator, Vector &state) {
    Vector weights = estimator(state);
    size_t a_star = Agent::argmaxQs(weights);
    // Epsilon-greedy policy
    for (size_t a = 0; a < actions; a++) {
        if (a == a_star)
            weights(a) = 1 - epsilon + epsilon / actions;
        else
            weights(a) = epsilon / actions;
    }
    return Agent::select_action(weights);
}

void ApproximateAgent::reset(size_t init_state) {
    Vector vector_state = {(float)init_state};
    reset(vector_state);
}

void ApproximateAgent::reset(std::list<float> init_state) { reset(to_xtensor(init_state)); }

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

void ApproximateAgent::update(std::list<float> next_state, float reward, bool done, py::kwargs kwargs) {
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
    out << "<" << demangled << "(epsilon=" << epsilon << ", gamma=" << gamma;
    out << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
