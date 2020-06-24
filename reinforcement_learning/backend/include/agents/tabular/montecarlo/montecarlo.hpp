#pragma once

#include <list>

#include "../../agent.ipp"

namespace rl {

namespace agents {

class MonteCarlo : public Agent {
   protected:
    Trajectory trajectory;
    xt::xtensor<float, 2> returns;

    virtual void control() = 0;

   public:
    MonteCarlo(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay);
    MonteCarlo(const MonteCarlo &other);
    virtual ~MonteCarlo();

    void reset(size_t init_state);
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace rl
