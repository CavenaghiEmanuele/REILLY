#pragma once

#include "double_expected_sarsa.hpp"

namespace rl {

namespace agents {

DoubleExpectedSarsa::DoubleExpectedSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : DoubleTemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

DoubleExpectedSarsa::DoubleExpectedSarsa(const DoubleExpectedSarsa &other) : DoubleTemporalDifference(other) {}

DoubleExpectedSarsa &DoubleExpectedSarsa::operator=(const DoubleExpectedSarsa &other) {
    if (this != &other) {
        DoubleExpectedSarsa tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.Q2, Q2);
        std::swap(tmp.pi, pi);
        std::swap(tmp.pi2, pi2);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
    }
    return *this;
}

DoubleExpectedSarsa::~DoubleExpectedSarsa() {}

void DoubleExpectedSarsa::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        float expected_value = 0;
        if (xt::random::binomial<int>({1})(0) == 0) {
            expected_value = xt::sum(xt::row(pi2, next_state) * xt::row(Q2, next_state))();
            Q(state, action) += alpha * (reward + gamma * expected_value - Q(state, action));
            policy_update(Q, pi, state);
        } else {
            expected_value = xt::sum(xt::row(pi, next_state) * xt::row(Q, next_state))();
            Q2(state, action) += alpha * (reward + gamma * expected_value - Q2(state, action));
            policy_update(Q2, pi2, state);
        }
    }

    state = next_state;
    action = select_action(pi + pi2, next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
