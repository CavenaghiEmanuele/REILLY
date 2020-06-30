#pragma once

#include "n_step_tree_backup.hpp"

namespace rl {

namespace agents {

NStepTreeBackup::NStepTreeBackup(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_step, float epsilon_decay)
    : NStep(states, actions, alpha, epsilon, gamma, n_step, epsilon_decay) {}

NStepTreeBackup::NStepTreeBackup(const NStepTreeBackup &other) : NStep(other) {}

NStepTreeBackup &NStepTreeBackup::operator=(const NStepTreeBackup &other) {
    if (this != &other) {
        NStepTreeBackup tmp(other);
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

NStepTreeBackup::~NStepTreeBackup() {}

void NStepTreeBackup::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    size_t t = py::cast<size_t>(kwargs["t"]);
    bool training = py::cast<bool>(kwargs["training"]);

    size_t next_action = select_action(pi, next_state);

    if (t < T) {
        trajectory.push_back({next_state, next_action, reward});
        if (done) T = t + 1;
    }

    if (training) {
        size_t tau = t - n_step + 1;
        if (t + 1 >= n_step) {  // Like tau >= 0, but for unsigned numbers
            Point *p;
            float G = 0;
            float expected_value = 0;

            if (t + 1 >= T) {
                G = trajectory[T].reward;
            } else {
                expected_value = xt::sum(xt::row(pi, next_state) * xt::row(Q, next_state))();
                G = reward + gamma * expected_value;
            }
            
            for (size_t k = std::min(t, T - 1); k >= tau + 1; k--) {
                p = &trajectory[k];
                expected_value = xt::sum(xt::row(pi, next_state) * xt::row(Q, next_state))();
                expected_value -= pi(p->state, p->action) * Q(p->state, p->action);
                G = p->reward + gamma * expected_value + gamma * pi(p->state, p->action) * G;
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
