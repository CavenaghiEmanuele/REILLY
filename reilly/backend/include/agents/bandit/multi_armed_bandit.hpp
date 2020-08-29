#pragma once

#include "../agent.ipp"

namespace reilly {

namespace agents {

template <typename Arm>
class MultiArmedBandit : public Agent {
   protected:
    virtual size_t select_action() = 0;

   public:
    MultiArmedBandit(size_t actions, float gamma = 1, float epsilon_decay = 1);
    MultiArmedBandit(const MultiArmedBandit &other);
    virtual ~MultiArmedBandit();

    std::vector<Arm> arms;

    virtual void reset(size_t init_state);
    virtual void update(size_t next_state, float reward, bool done, py::kwargs kwargs);

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace reilly
