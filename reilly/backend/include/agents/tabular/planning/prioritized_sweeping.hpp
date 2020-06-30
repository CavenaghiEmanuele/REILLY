#pragma once

#include "q_planning.ipp"

namespace rl {

namespace agents {

class PrioritizedSweeping : public QPlanning {
   protected:
    float theta;
    std::priority_queue<Point> pqueue;

   public:
    PrioritizedSweeping(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan,
                        float theta, float epsilon_decay = 1);
    PrioritizedSweeping(const PrioritizedSweeping &other);
    PrioritizedSweeping &operator==(const PrioritizedSweeping &other);
    ~PrioritizedSweeping();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl
