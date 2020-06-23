#pragma once

#include "double_q_learning.hpp"

namespace rl {

namespace agents {

DoubleQLearning::DoubleQLearning(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay)
    : DoubleTemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay) {}

DoubleQLearning::DoubleQLearning(const DoubleQLearning &other) : DoubleTemporalDifference(other) {}

DoubleQLearning &DoubleQLearning::operator=(const DoubleQLearning &other) {
    if (this != &other) {
        DoubleQLearning tmp(other);
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

DoubleQLearning::~DoubleQLearning() {}

void DoubleQLearning::update(size_t next_state, float reward, bool done, bool training) {
    if (training) {
        if (xt::random::binomial<int>({1})(0) == 0) {
            Q(state, action) += alpha * (reward + gamma * xt::amax(xt::row(Q2, next_state))() - Q(state, action));
            policy_update(Q, pi, state);
        } else {
            Q2(state, action) += alpha * (reward + gamma * xt::amax(xt::row(Q, next_state))() - Q2(state, action));
            policy_update(Q2, pi2, state);
        }
    }

    state = next_state;
    action = select_action(pi + pi2, next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
