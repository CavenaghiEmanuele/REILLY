#pragma once

#include "../agent.ipp"

namespace reilly {

namespace agents {

using ActionValue = xt::xtensor<float, 2>;
using Policy = xt::xtensor<float, 2>;

class TabularAgent : public Agent {
   protected:
    size_t states;

    ActionValue Q;
    Policy pi;

    size_t state;
    
    inline virtual size_t select_action(const Policy &pi, size_t state);
    inline virtual void policy_update(const ActionValue &Q, Policy &pi, size_t state);

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
    TabularAgent(size_t states, size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    TabularAgent(const TabularAgent &other);
    virtual ~TabularAgent();

    std::string __repr__();
};

}  // namespace agents

}  // namespace reilly
