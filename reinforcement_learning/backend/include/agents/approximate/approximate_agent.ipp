#pragma once

#include "approximate_agent.hpp"

namespace rl {

namespace agents {

ApproximateAgent::ApproximateAgent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay,
                                   size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size)
    : Agent(actions, alpha, epsilon, gamma, epsilon_decay),
      estimator(alpha, tilings, to_xtensor(tilings_offset), to_xtensor(tile_size)) {}

ApproximateAgent::ApproximateAgent(const ApproximateAgent &other) : Agent(other), estimator(other.estimator) {}

ApproximateAgent::~ApproximateAgent() {}

inline size_t ApproximateAgent::select_action(TileCoding &estimator, State &state) {
    xt::xtensor<float, 1> weights = xt::empty<float>({actions});
    for (size_t a = 0; a < actions; a++) weights(a) = estimator(state, a);
    // Argmax breaking ties arbitrarily
    size_t a_star = xt::random::choice(
        xt::ravel_indices(xt::argwhere(xt::equal(weights, xt::amax(weights))), weights.shape()), 1)(0);
    // Epsilon-greedy policy
    for (size_t a = 0; a < actions; a++) {
        if (a == a_star) {
            weights(a) = 1 - epsilon + epsilon / actions;
        } else {
            weights(a) = epsilon / actions;
        }
    }
    std::discrete_distribution<size_t> distribution(weights.cbegin(), weights.cend());
    return distribution(generator);
}

void ApproximateAgent::reset(size_t init_state) {
    State vector_state = {(float)init_state};
    reset(vector_state);
}

void ApproximateAgent::reset(std::list<float> init_state) { reset(to_xtensor(init_state)); }

void ApproximateAgent::reset(py::array init_state) {
    auto np = init_state.unchecked<float, 1>();
    State vector_state = xt::zeros<float>({np.size()});
    for (long int i = 0; i < np.size(); i++) {
        vector_state[i] = np(i);
    }
    reset(vector_state);
}

void ApproximateAgent::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    State vector_state = {(float)next_state};
    update(vector_state, reward, done, kwargs);
}

void ApproximateAgent::update(std::list<float> next_state, float reward, bool done, py::kwargs kwargs) {
    update(to_xtensor(next_state), reward, done, kwargs);
}

void ApproximateAgent::update(py::array next_state, float reward, bool done, py::kwargs kwargs) {
    auto np = next_state.unchecked<float, 1>();
    State vector_state = xt::zeros<float>({np.size()});
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
