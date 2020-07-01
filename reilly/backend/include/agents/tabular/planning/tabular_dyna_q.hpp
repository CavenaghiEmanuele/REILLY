#pragma once

#include "q_planning.ipp"

namespace reilly {

namespace agents {

class TabularDynaQ : public QPlanning {
   public:
    TabularDynaQ(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float epsilon_decay = 1);
    TabularDynaQ(const TabularDynaQ &other);
    TabularDynaQ &operator==(const TabularDynaQ &other);
    ~TabularDynaQ();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
