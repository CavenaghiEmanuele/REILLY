#pragma once

#include "n_step_sarsa.hpp"

namespace rl {

namespace agents {

NStepSarsa::NStepSarsa(size_t states, size_t actions, float alpha, float epsilon, float gamma, int64_t n_step, float epsilon_decay)
    : NStep(states, actions, alpha, epsilon, gamma, n_step, epsilon_decay) {}

NStepSarsa::NStepSarsa(const NStepSarsa &other) : NStep(other) {}

NStepSarsa &NStepSarsa::operator=(const NStepSarsa &other) {
    if (this != &other) {
        NStepSarsa tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.pi, pi);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.n_step, n_step);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
        std::swap(tmp.T, T);
        std::swap(tmp.trajectory, trajectory);
    }
    return *this;
}

NStepSarsa::~NStepSarsa() {}

void NStepSarsa::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    int64_t t = py::cast<int64_t>(kwargs["t"]);
    bool training = py::cast<bool>(kwargs["training"]);

    size_t next_action = select_action(pi, next_state);

    if (t < T) {
        trajectory.push_back({next_state, next_action, reward});
        if (done) T = t + 1;
    }

    if (training) {
        int64_t tau = t - n_step + 1;
        if (tau >= 0) {
            Point *p;
            float G = 0;
            
            for (int64_t i = tau + 1; i < std::min(T, tau + n_step); i++) {
                G += std::pow(gamma, i - tau - 1) * trajectory[i].reward;
            }

            if (tau + n_step < T) {
                p = &trajectory[tau + n_step];
                G += std::pow(gamma, n_step) * Q(p->state, p->action);
            }

            p = &trajectory[tau];
            Q(p->state, p->action) += alpha * (G - Q(p->state, p->action));
            policy_update(Q, pi, p->state);
        }
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace rl
