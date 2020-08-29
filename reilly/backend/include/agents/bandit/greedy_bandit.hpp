#pragma once

#include "multi_armed_bandit.ipp"

namespace reilly {

namespace agents {

template <typename Arm>
class GreedyBandit : public MultiArmedBandit<Arm> {
   protected:
    size_t select_action();

   public:
    GreedyBandit(size_t actions, float gamma = 1, float epsilon_decay = 1);
    GreedyBandit(const GreedyBandit &other);
    GreedyBandit &operator=(const GreedyBandit &other);
    virtual ~GreedyBandit();
};

}  // namespace agents

}  // namespace reilly
