#pragma once

#include "thompson_sampling_bandit.hpp"

namespace reilly {

namespace agents {

template <typename Arm>
ThompsonSamplingBandit<Arm>::ThompsonSamplingBandit(size_t actions, float epsilon_decay) : MultiArmedBandit<Arm>(actions, epsilon_decay) {}

template <typename Arm>
ThompsonSamplingBandit<Arm>::ThompsonSamplingBandit(const ThompsonSamplingBandit &other) : MultiArmedBandit<Arm>(other) {}

template <typename Arm>
ThompsonSamplingBandit<Arm> &ThompsonSamplingBandit<Arm>::operator=(const ThompsonSamplingBandit &other) {
    if (this != &other) {
        ThompsonSamplingBandit tmp(other);
        std::swap(other.actions, this->actions);
        std::swap(other.epsilon_decay, this->epsilon_decay);
        std::swap(other.arms, this->arms);
    }
    return *this;
}

template <typename Arm>
ThompsonSamplingBandit<Arm>::~ThompsonSamplingBandit() {}

template <typename Arm>
size_t ThompsonSamplingBandit<Arm>::select_action() {
    std::vector<float> weights;
    for (Arm arm : this->arms) weights.push_back(arm.sample());
    Vector out = to_xtensor(weights);
    return this->argmaxQs(out);
}

}  // namespace agents

}  // namespace reilly
