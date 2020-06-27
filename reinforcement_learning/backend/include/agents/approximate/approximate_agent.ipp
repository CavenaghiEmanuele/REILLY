#pragma once

#include "approximate_agent.hpp"

namespace rl {

namespace agents {

ApproximateAgent::ApproximateAgent(float alpha, float epsilon, float gamma, float epsilon_decay)
    : Agent(alpha, epsilon, gamma, epsilon_decay) {}

ApproximateAgent::ApproximateAgent(const ApproximateAgent &other) : Agent(other) {}

ApproximateAgent::~ApproximateAgent() {}

void ApproximateAgent::reset(size_t init_state) {
    State vector_state = {(float) init_state};
    reset(vector_state);
}

void ApproximateAgent::reset(std::list<float> init_state) {
    size_t i = 0;
    State vector_state = xt::zeros<float>({init_state.size()});
    for (std::list<float>::iterator j = init_state.begin(); j != init_state.end(); j++) {
        vector_state[i] = *j;
        i++;
    }
    reset(vector_state);
}

void ApproximateAgent::reset(py::array init_state) {
    auto np = init_state.unchecked<float, 1>();
    State vector_state = xt::zeros<float>({np.size()});
    for (long int i = 0; i < np.size(); i++) {
        vector_state[i] = np(i);
    }
    reset(vector_state);
}

void ApproximateAgent::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    State vector_state = {(float) next_state};
    update(vector_state, reward, done, kwargs);
}

void ApproximateAgent::update(std::list<float> next_state, float reward, bool done, py::kwargs kwargs) {
    size_t i = 0;
    State vector_state = xt::zeros<float>({next_state.size()});
    for (std::list<float>::iterator j = next_state.begin(); j != next_state.end(); j++) {
        vector_state[i] = *j;
        i++;
    }
    update(vector_state, reward, done, kwargs);
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
