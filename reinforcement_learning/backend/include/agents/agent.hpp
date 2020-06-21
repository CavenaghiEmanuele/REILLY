#pragma once

#include <cxxabi.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace rl {

namespace agents {

using ActionValue = xt::xtensor<float, 2>;
using StateValue = xt::xtensor<float, 1>;
using Policy = xt::xtensor<float, 2>;

class Agent {
   protected:
    size_t states;
    size_t actions;

    ActionValue Q;
    Policy pi;

    float epsilon;
    float gamma;

    float epsilon_decay;

    size_t state;
    size_t action;

    std::default_random_engine generator;

   public:
    Agent(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay = 1);
    Agent(const Agent &other);
    virtual ~Agent();

    virtual size_t get_action();
    virtual void reset(size_t init_state) = 0;
    virtual void update(size_t next_state, float reward, bool done, bool training = true) = 0;

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace rl
