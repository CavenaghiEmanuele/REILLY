#pragma once

#include <cxxabi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <queue>
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

class Agent {
   protected:
    float alpha;
    float epsilon;
    float gamma;

    float epsilon_decay;

    size_t state;
    size_t action;

    std::minstd_rand generator;

    struct Point {
        size_t state;
        size_t action;
        float reward;

        bool operator==(const Point &other) const { return state == other.state && action == other.action; }
        bool operator!=(const Point &other) const { return !(this == &other); }
        bool operator<(const Point &other) const { return reward < other.reward; }
    };

    using Trajectory = std::vector<Point>;

   public:
    Agent(float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    Agent(const Agent &other);
    virtual ~Agent();

    virtual size_t get_action();
    virtual void reset(size_t init_state) = 0;
    virtual void update(size_t next_state, float reward, bool done, py::kwargs kwargs) = 0;

    virtual std::string __repr__() = 0;
};

}  // namespace agents

}  // namespace rl
