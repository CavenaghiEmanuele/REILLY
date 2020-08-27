#pragma once

#include "../agent.ipp"
#include "tile_coding.ipp"

namespace reilly {

namespace agents {

class ApproximateAgent : public Agent {
   protected:
    Vector state;
    TileCoding estimator;

    inline virtual size_t select_action(TileCoding &estimator, Vector &state);

    virtual void reset(Vector init_state) = 0;
    virtual void update(Vector next_state, float reward, bool done, py::kwargs kwargs) = 0;

    struct Point {
        Vector state;
        size_t action;
        float reward;
    };

    using Trajectory = std::vector<Point>;

   public:
    ApproximateAgent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, py::kwargs kwargs);
    ApproximateAgent(const ApproximateAgent &other);
    virtual ~ApproximateAgent();

    void reset(size_t init_state);
    void reset(std::vector<float> init_state);
    void reset(py::array init_state);

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
    void update(std::vector<float> next_state, float reward, bool done, py::kwargs kwargs);
    void update(py::array next_state, float reward, bool done, py::kwargs kwargs);

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace reilly
