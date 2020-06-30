#pragma once

#include "every_visit.hpp"

namespace rl {

namespace agents {

MonteCarloEveryVisit::MonteCarloEveryVisit(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : MonteCarlo(states, actions, epsilon, gamma, epsilon_decay) {}

MonteCarloEveryVisit::MonteCarloEveryVisit(const MonteCarloEveryVisit &other) : MonteCarlo(other) {}

MonteCarloEveryVisit &MonteCarloEveryVisit::operator=(const MonteCarloEveryVisit &other) {
    if (this != &other) {
        MonteCarloEveryVisit tmp(other);
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
        std::swap(tmp.trajectory, trajectory);
        std::swap(tmp.returns, returns);
    }
    return *this;
}

MonteCarloEveryVisit::~MonteCarloEveryVisit() {}

void MonteCarloEveryVisit::control() {
    float G = 0;
    Trajectory::reverse_iterator t;
    // For each step of episode, t = T-1, T-2, ..., 0
    for (t = trajectory.rbegin(); t != trajectory.rend(); ++t) {
        G = t->reward + gamma * G;
        // Append G to Returns(St, At)
        returns(t->state, t->action) += 1;
        // Incremental update of Average(Returns(St, At))
        Q(t->state, t->action) += (G - Q(t->state, t->action)) / returns(t->state, t->action);
        // Update policy
        policy_update(Q, pi, t->state);
    }
}

}  // namespace agents

}  // namespace rl
