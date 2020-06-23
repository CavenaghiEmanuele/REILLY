#pragma once

#include "montecarlo.hpp"

namespace rl {

namespace agents {

MonteCarlo::MonteCarlo(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay)
    : Agent(states, actions, epsilon, gamma, epsilon_decay) {
    returns = xt::zeros<float>({states, actions});
}

MonteCarlo::MonteCarlo(const MonteCarlo &other) : Agent(other), trajectory(other.trajectory), returns(other.returns) {}

MonteCarlo::~MonteCarlo() {}

size_t MonteCarlo::get_action() {
    action = select_action(state);
    return action;
}

void MonteCarlo::reset(size_t init_state) {
    state = init_state;
    trajectory.clear();
}

void MonteCarlo::update(size_t next_state, float reward, bool done, bool training) {
    trajectory.push_back({state, action, reward});
    if (done) {
        if (training) control();
        epsilon *= epsilon_decay;
    }
    state = next_state;
}

}  // namespace agents

}  // namespace rl
