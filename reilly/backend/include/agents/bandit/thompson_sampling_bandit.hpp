#pragma once

#include "multi_armed_bandit.ipp"

namespace reilly {

namespace agents {

template <typename Arm>
class ThompsonSamplingBandit : public MultiArmedBandit<Arm> {
   protected:
    size_t select_action();

   public:
    ThompsonSamplingBandit(size_t actions, float gamma = 1, float epsilon_decay = 1);
    ThompsonSamplingBandit(const ThompsonSamplingBandit &other);
    ThompsonSamplingBandit &operator=(const ThompsonSamplingBandit &other);
    virtual ~ThompsonSamplingBandit();
};

}  // namespace agents

}  // namespace reilly
