#pragma once

#include "../agent.ipp"

namespace rl {

namespace agents {

using State = xt::xtensor<float, 1>;

class ApproximateAgent : public Agent {
   protected:
    State state;

   public:
    ApproximateAgent(float alpha, float epsilon, float gamma, float epsilon_decay = 1);
    ApproximateAgent(const ApproximateAgent &other);
    virtual ~ApproximateAgent();

    void reset(size_t init_state);
    void reset(std::list<float> init_state);
    void reset(py::array init_state);

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
    void update(std::list<float> next_state, float reward, bool done, py::kwargs kwargs);
    void update(py::array next_state, float reward, bool done, py::kwargs kwargs);

    virtual void reset(State init_state) = 0;
    virtual void update(State next_state, float reward, bool done, py::kwargs kwargs) = 0;

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace rl
