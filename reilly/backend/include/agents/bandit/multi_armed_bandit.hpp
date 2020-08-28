#pragma once

#include "../agent.ipp"

namespace reilly {

namespace agents {

struct BernoulliArm {
    size_t alpha = 1;
    size_t beta = 1;
    std::vector<float> trace{0.5};

    size_t taken = 0;

    void update(float reward, float decay) {
        taken++;
        alpha += 1 * (reward > 0);
        beta += 1 * (reward <= 0);
    }

    template<class Generator>
    float sample(Generator &generator) {
        // X = Gamma(alpha, 1), Y = Gamma(beta, 1) ->
        // Z = X / X+Y = Beta(alpha, beta)
        std::gamma_distribution<> X(alpha, 1);
        std::gamma_distribution<> Y(beta, 1);
        float x = X(generator); float y = Y(generator);
        return x / (x+y);
    }

    operator float() const { return (float)alpha / (float)(alpha + beta); }
};

struct GaussianArm {
    float mu = 0.5;
    float stddev = 1;
    std::vector<float> trace{0.5};

    size_t taken = 0;

    void update(float reward, float decay) {
        taken++;
        stddev *= decay;
        mu += (1 / (float) taken) * (reward - mu);
    }

    template<class Generator>
    float sample(Generator &generator) {
        std::normal_distribution<> normal(mu, stddev);
        return normal(generator);
    }

    operator float() const { return mu; }
};

template <typename Arm>
class MultiArmedBandit : public Agent {
   protected:
    virtual size_t select_action() = 0;

   public:
    MultiArmedBandit(size_t actions, float epsilon_decay = 1);
    MultiArmedBandit(const MultiArmedBandit &other);
    virtual ~MultiArmedBandit();

    std::vector<Arm> arms;

    virtual void reset(size_t init_state);
    virtual void update(size_t next_state, float reward, bool done, py::kwargs kwargs);

    virtual std::string __repr__();
};

}  // namespace agents

}  // namespace reilly
