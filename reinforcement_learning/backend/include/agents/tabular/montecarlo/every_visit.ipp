#pragma once

#include "every_visit.hpp"

namespace rl {

namespace agents {

MonteCarloEveryVisit::MonteCarloEveryVisit(size_t states, size_t actions, float epsilon, float gamma,
                                           float epsilon_decay)
    : MonteCarlo(states, actions, epsilon, gamma, epsilon_decay) {}

MonteCarloEveryVisit::MonteCarloEveryVisit(const MonteCarloEveryVisit &other) : MonteCarlo(other) {}

MonteCarloEveryVisit &MonteCarloEveryVisit::operator=(const MonteCarloEveryVisit &other) {
    if (this != &other) {
        MonteCarloEveryVisit tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.pi, pi);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
        std::swap(tmp.trajectory, trajectory);
        std::swap(tmp.returns, returns);
    }
    return *this;
}

MonteCarloEveryVisit::~MonteCarloEveryVisit() {}

void MonteCarloEveryVisit::control() {
    float G = 0, Q_star = 0;
    Trajectory::reverse_iterator t;
    // For each step of episode, t = T-1, T-2, ..., 0
    for (t = trajectory.rbegin(); t != trajectory.rend(); ++t) {
        G = gamma * G + t->reward;
        // Append G to Returns(St, At)
        returns(t->state, t->action) += 1;
        // Incremental update of Average(Returns(St, At))
        Q_star = (G - Q(t->state, t->action)) / returns(t->state, t->action);
        Q(t->state, t->action) += Q_star;
        // Update policy
        policy_update(t->state);
    }
}

}  // namespace agents

}  // namespace rl
