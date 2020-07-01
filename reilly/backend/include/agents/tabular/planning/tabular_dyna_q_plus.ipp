#pragma once

#include "tabular_dyna_q_plus.hpp"

namespace reilly {

namespace agents {

TabularDynaQPlus::TabularDynaQPlus(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float epsilon_decay)
    : QPlanning(states, actions, alpha, epsilon, gamma, n_plan, epsilon_decay) {}

TabularDynaQPlus::TabularDynaQPlus(const TabularDynaQPlus &other) : QPlanning(other) {}

TabularDynaQPlus &TabularDynaQPlus::operator==(const TabularDynaQPlus &other) {
    if (this != &other) {
        TabularDynaQPlus tmp(other);
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
        std::swap(tmp.n_plan, n_plan);
        std::swap(tmp.model, model);
    }
    return *this;
}

TabularDynaQPlus::~TabularDynaQPlus() {}

void TabularDynaQPlus::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        Q(state, action) += alpha * (reward + gamma * xt::amax(xt::row(Q, next_state))() - Q(state, action));
        policy_update(Q, pi, state);
        model.set_result(state, action, reward, next_state);
        for (size_t i = 0; i < n_plan; i++) {
            size_t prev_state = model.get_random_observed_state();
            size_t prev_action = (size_t) generator() % actions;
            Result r = model(prev_state, prev_action);
            Q(prev_state, prev_action) += alpha * (r.reward + gamma * xt::amax(xt::row(Q, r.next_state))() - Q(prev_state, prev_action));
            policy_update(Q, pi, prev_state);
        }
    }

    state = next_state;
    action = select_action(pi, next_state);

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace reilly
