#pragma once

#include "multi_armed_bandit.ipp"

namespace reilly {

namespace agents {

template <typename Arm>
class UCBBandit : public MultiArmedBandit<Arm> {
   protected:
    size_t select_action();

   public:
    UCBBandit(size_t actions, float gamma = 1, float epsilon_decay = 1);
    UCBBandit(const UCBBandit &other);
    UCBBandit &operator=(const UCBBandit &other);
    virtual ~UCBBandit();
};

}  // namespace agents

}  // namespace reilly
