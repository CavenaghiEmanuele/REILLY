#pragma once

#include "first_visit.hpp"

namespace reilly {

namespace agents {

MonteCarloFirstVisit::MonteCarloFirstVisit(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : MonteCarlo(states, actions, epsilon, gamma, epsilon_decay) {}

MonteCarloFirstVisit::MonteCarloFirstVisit(const MonteCarloFirstVisit &other) : MonteCarlo(other) {}

MonteCarloFirstVisit &MonteCarloFirstVisit::operator=(const MonteCarloFirstVisit &other) {
    if (this != &other) {
        MonteCarloFirstVisit tmp(other);
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

MonteCarloFirstVisit::~MonteCarloFirstVisit() {}

void MonteCarloFirstVisit::control() {
    float G = 0;
    Trajectory::reverse_iterator t;
    // For each step of episode, t = T-1, T-2, ..., 0
    for (t = trajectory.rbegin(); t != trajectory.rend(); ++t) {
        G = gamma * G + t->reward;
        // Unless the pair St, At appears in S0, A0, ..., St-1, At-1
        Trajectory::iterator end = (t+1).base();
        if (std::find(trajectory.begin(), end, *t) == end) {
            // Append G to Returns(St, At)
            returns(t->state, t->action) += 1;
            // Incremental update of Average(Returns(St, At))
            Q(t->state, t->action) += (G - Q(t->state, t->action)) / returns(t->state, t->action);
            // Update policy
            policy_update(Q, pi, t->state);
        }
    }
}

}  // namespace agents

}  // namespace reilly
