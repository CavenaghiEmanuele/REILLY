#pragma once

#include "q_learning.hpp"

namespace rl {

namespace agents {

QLearning::QLearning(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

QLearning::QLearning(const QLearning &other) : TemporalDifference(other) {}

QLearning &QLearning::operator=(const QLearning &other) {
    if (this != &other) {
        QLearning tmp(other);
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

QLearning::~QLearning() {}

void QLearning::update(size_t next_state, float reward, bool done, bool training) {
    if (training) {
        Q(state, action) += alpha * (reward + gamma * xt::amax(xt::row(Q, next_state))() - Q(state, action));
        policy_update(state);
    }

    state = next_state;
    action = select_action(next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
