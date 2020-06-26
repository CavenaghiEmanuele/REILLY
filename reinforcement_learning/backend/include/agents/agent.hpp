#pragma once

#include <cxxabi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace py = pybind11;

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

    float alpha;
    float epsilon;
    float gamma;

    float epsilon_decay;

    size_t state;
    size_t action;

    std::minstd_rand generator;

    inline size_t argmaxQs(const ActionValue &Q, size_t state);
    inline virtual size_t select_action(const Policy &pi, size_t state);
    inline virtual void policy_update(const ActionValue &Q, Policy &pi, size_t state);

    struct Point {
        size_t state;
        size_t action;
        float reward;

        bool operator==(const Point &other) { return state == other.state && action == other.action; }
        bool operator!=(const Point &other) { return !(this == &other); }
    };

    using Trajectory = std::vector<Point>;

   public:
    Agent(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    Agent(const Agent &other);
    virtual ~Agent();

    virtual size_t get_action();
    virtual void reset(size_t init_state) = 0;
    virtual void update(size_t next_state, float reward, bool done, py::kwargs kwargs) = 0;

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace rl
