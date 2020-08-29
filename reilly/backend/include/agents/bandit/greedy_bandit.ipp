#pragma once

#include "greedy_bandit.hpp"

namespace reilly {

namespace agents {

template <typename Arm>
GreedyBandit<Arm>::GreedyBandit(size_t actions, float gamma, float epsilon_decay) : MultiArmedBandit<Arm>(actions, gamma, epsilon_decay) {}

template <typename Arm>
GreedyBandit<Arm>::GreedyBandit(const GreedyBandit &other) : MultiArmedBandit<Arm>(other) {}

template <typename Arm>
GreedyBandit<Arm> &GreedyBandit<Arm>::operator=(const GreedyBandit &other) {
    if (this != &other) {
        GreedyBandit tmp(other);
        std::swap(tmp.actions, this->actions);
        std::swap(tmp.gamma, this->gamma);
        std::swap(tmp.epsilon_decay, this->epsilon_decay);
        std::swap(tmp.arms, this->arms);
    }
    return *this;
}

template <typename Arm>
GreedyBandit<Arm>::~GreedyBandit() {}

template <typename Arm>
size_t GreedyBandit<Arm>::select_action() {
    std::vector<float> weights;
    for (Arm arm : this->arms) weights.push_back((float) arm);
    Vector out = to_xtensor(weights);
    return this->argmaxQs(out);
}

}  // namespace agents

}  // namespace reilly
