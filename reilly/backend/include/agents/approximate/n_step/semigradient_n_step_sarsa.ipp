#pragma once

#include "semigradient_n_step_sarsa.hpp"

namespace reilly {

namespace agents {

SemiGradientNStepSarsa::SemiGradientNStepSarsa(size_t actions, float alpha, float epsilon, float gamma, size_t n_step,
                                               float epsilon_decay, py::kwargs kwargs)
    : ApproximateNStep(actions, alpha, epsilon, gamma, n_step, epsilon_decay, kwargs) {}

SemiGradientNStepSarsa::SemiGradientNStepSarsa(const SemiGradientNStepSarsa &other) : ApproximateNStep(other) {}

SemiGradientNStepSarsa &SemiGradientNStepSarsa::operator=(const SemiGradientNStepSarsa &other) {
    if (this != &other) {
        SemiGradientNStepSarsa tmp(other);
        std::swap(tmp.action, action);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.estimator, estimator);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
        std::swap(tmp.n_step, n_step);
        std::swap(tmp.T, T);
        std::swap(tmp.trajectory, trajectory);
    }
    return *this;
}

SemiGradientNStepSarsa::~SemiGradientNStepSarsa() {}

void SemiGradientNStepSarsa::update(Vector next_state, float reward, bool done, py::kwargs kwargs) {
    size_t t = py::cast<size_t>(kwargs["t"]);
    bool training = py::cast<bool>(kwargs["training"]);

    size_t next_action = select_action(estimator, next_state);

    if (t < T) {
        trajectory.push_back({next_state, next_action, reward});
        if (done) T = t + 1;
    }

    if (training) {
        size_t tau = t - n_step + 1;
        if (t + 1 >= n_step) {  // Like tau >= 0, but for unsigned numbers
            Point *p;
            float G = 0;

            for (size_t i = tau + 1; i < std::min(T, tau + n_step); i++) {
                G += std::pow(gamma, i - tau - 1) * trajectory[i].reward;
            }

            if (tau + n_step < T) {
                p = &trajectory[tau + n_step];
                G += std::pow(gamma, n_step) * estimator(p->state, p->action);
            }

            p = &trajectory[tau];
            estimator.update(p->state, p->action, G);
        }
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace reilly