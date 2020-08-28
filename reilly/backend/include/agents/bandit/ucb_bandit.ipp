#pragma once

#include "ucb_bandit.hpp"

namespace reilly {

namespace agents {

template <typename Arm>
UCBBandit<Arm>::UCBBandit(size_t actions, float epsilon_decay) : MultiArmedBandit<Arm>(actions, epsilon_decay) {}

template <typename Arm>
UCBBandit<Arm>::UCBBandit(const UCBBandit &other) : MultiArmedBandit<Arm>(other) {}

template <typename Arm>
UCBBandit<Arm> &UCBBandit<Arm>::operator=(const UCBBandit &other) {
    if (this != &other) {
        UCBBandit tmp(other);
        std::swap(other.actions, this->actions);
        std::swap(other.epsilon_decay, this->epsilon_decay);
        std::swap(other.arms, this->arms);
    }
    return *this;
}

template <typename Arm>
UCBBandit<Arm>::~UCBBandit() {}

template <typename Arm>
size_t UCBBandit<Arm>::select_action() {
    float taken = 0;
    std::vector<float> weights;
    for (Arm arm : this->arms) taken += arm.taken;
    for (Arm arm : this->arms) {
        float ucb = std::numeric_limits<float>::infinity();
        if (taken != 0 && arm.taken != 0) ucb = 2 * sqrt(log(taken) / (float) arm.taken);
        weights.push_back((float) arm + ucb);
    }
    Vector out = to_xtensor(weights);
    return this->argmaxQs(out);
}

}  // namespace agents

}  // namespace reilly
