#pragma once

#include "semigradient_montecarlo.hpp"

namespace rl {

namespace agents {

SemiGradientMonteCarlo::SemiGradientMonteCarlo(size_t actions, float alpha, float epsilon, float gamma,
                                               float epsilon_decay, py::kwargs kwargs)
    : ApproximateAgent(actions, alpha, epsilon, gamma, epsilon_decay, kwargs) {}

SemiGradientMonteCarlo::SemiGradientMonteCarlo(const SemiGradientMonteCarlo &other)
    : ApproximateAgent(other), trajectory(other.trajectory) {}

SemiGradientMonteCarlo &SemiGradientMonteCarlo::operator=(const SemiGradientMonteCarlo &other) {
    if (this != &other) {
        SemiGradientMonteCarlo tmp(other);
        std::swap(tmp.action, action);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.estimator, estimator);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
        std::swap(tmp.trajectory, trajectory);
    }
    return *this;
}

SemiGradientMonteCarlo::~SemiGradientMonteCarlo() {}

void SemiGradientMonteCarlo::reset(Vector init_state) {
    state = init_state;
    action = select_action(estimator, init_state);
    trajectory.clear();
}

void SemiGradientMonteCarlo::update(Vector next_state, float reward, bool done, py::kwargs kwargs) {
    trajectory.push_back({state, action, reward});

    bool training = py::cast<bool>(kwargs["training"]);
    if (done && training) {
        float G = 0;
        for (auto t : trajectory) {
            G = t.reward + gamma * G;
            estimator.update(t.state, t.action, G);
        }
    }

    state = next_state;
    action = select_action(estimator, next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl