#pragma once

#include "double_sarsa.hpp"

namespace rl {

namespace agents {

DoubleSarsa::DoubleSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : DoubleTemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

DoubleSarsa::DoubleSarsa(const DoubleSarsa &other) : DoubleTemporalDifference(other) {}

DoubleSarsa &DoubleSarsa::operator=(const DoubleSarsa &other) {
    if (this != &other) {
        DoubleSarsa tmp(other);
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

DoubleSarsa::~DoubleSarsa() {}

void DoubleSarsa::update(size_t next_state, float reward, bool done, bool training) {
    size_t next_action = select_action(pi + pi2, next_state);

    if (training) {
        if (xt::random::binomial<int>({1})(0) == 0) {
            Q(state, action) += alpha * (reward + gamma * Q2(next_state, next_action) - Q(state, action));
            policy_update(Q, pi, state);
        } else {
            Q2(state, action) += alpha * (reward + gamma * Q(next_state, next_action) - Q2(state, action));
            policy_update(Q2, pi2, state);
        }
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
