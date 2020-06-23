#pragma once

#include "expected_sarsa.hpp"

namespace rl {

namespace agents {

ExpectedSarsa::ExpectedSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

ExpectedSarsa::ExpectedSarsa(const ExpectedSarsa &other) : TemporalDifference(other) {}

ExpectedSarsa &ExpectedSarsa::operator=(const ExpectedSarsa &other) {
    if (this != &other) {
        ExpectedSarsa tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.pi, pi);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
    }
    return *this;
}

ExpectedSarsa::~ExpectedSarsa() {}

void ExpectedSarsa::update(size_t next_state, float reward, bool done, bool training) {
    if (training) {
        float expected_value = xt::sum(xt::row(pi, next_state) * xt::row(Q, next_state))();
        Q(state, action) += alpha * (reward + gamma * expected_value - Q(state, action));
        policy_update(state);
    }

    state = next_state;
    action = select_action(next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
