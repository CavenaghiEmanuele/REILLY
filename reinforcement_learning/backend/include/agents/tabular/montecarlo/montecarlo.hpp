#pragma once

#include <list>

#include "../../agent.ipp"

namespace rl {

namespace agents {

class MonteCarlo : public Agent {
   protected:
    struct Point {
        size_t state;
        size_t action;
        float reward;

        bool operator==(const Point &other) {
            return state == other.state && action == other.action;
        }

        bool operator!=(const Point &other) {
            return !(this == &other); 
        }
    };

    using Trajectory = std::vector<Point>;

    Trajectory trajectory;
    xt::xtensor<float, 2> returns;

    virtual void control() = 0;

   public:
    MonteCarlo(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay);
    MonteCarlo(const MonteCarlo &other);
    virtual ~MonteCarlo();

    void reset(size_t init_state);
    void update(size_t next_state, float reward, bool done, bool training);
};

}  // namespace agents

}  // namespace rl
