#pragma once

#include "../agent.ipp"
#include "tile_coding.ipp"

namespace rl {

namespace agents {

Vector to_xtensor(std::list<float> in) {
    size_t i = 0;
    Vector out = xt::empty<float>({in.size()});
    for (auto j = in.begin(); j != in.end(); j++) {
        out[i] = (float)(*j);
        i++;
    }
    return out;
}

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
    ApproximateAgent(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, size_t features,
                     size_t tilings, std::list<float> tilings_offset, std::list<float> tile_size);
    ApproximateAgent(const ApproximateAgent &other);
    virtual ~ApproximateAgent();

    void reset(size_t init_state);
    void reset(std::list<float> init_state);
    void reset(py::array init_state);

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs);
    void update(std::list<float> next_state, float reward, bool done, py::kwargs kwargs);
    void update(py::array next_state, float reward, bool done, py::kwargs kwargs);

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace rl
