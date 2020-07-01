#pragma once

#include "q_planning.ipp"

namespace reilly {

namespace agents {

class TabularDynaQPlus : public QPlanning {
   public:
    TabularDynaQPlus(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float epsilon_decay = 1);
    TabularDynaQPlus(const TabularDynaQPlus &other);
    TabularDynaQPlus &operator==(const TabularDynaQPlus &other);
    ~TabularDynaQPlus();

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
};

}  // namespace agents

}  // namespace reilly
