#pragma once

#include <cxxabi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace py = pybind11;

namespace reilly {

namespace agents {

using Vector = xt::xtensor<float, 1>;

Vector to_xtensor(std::vector<float> other) {
    Vector out = xt::adapt(other, {other.size()});
    return out;
}

class Agent {
   protected:
    size_t actions;

    float alpha;
    float epsilon;
    float gamma;

    float epsilon_decay;

    size_t action;

    std::minstd_rand generator;

    inline size_t argmaxQs(Vector &weights);
    inline Vector e_greedy_policy(Vector &weights);
    inline size_t select_action(Vector &weights);

   public:
    Agent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    Agent(const Agent &other);
    virtual ~Agent();

    virtual size_t get_action();
    virtual void reset(size_t init_state) = 0;
    virtual void update(size_t next_state, float reward, bool done, py::kwargs kwargs) = 0;

    virtual std::string __repr__() = 0;
};

}  // namespace agents

}  // namespace reilly
