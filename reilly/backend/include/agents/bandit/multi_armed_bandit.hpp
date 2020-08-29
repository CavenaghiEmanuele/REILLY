#pragma once

#include "../agent.ipp"

namespace reilly {

namespace agents {

struct BernoulliArm {
    size_t alpha = 2;
    size_t beta = 2;
    std::vector<float> trace{0.5};

    size_t count = 0;

    void update(float reward, float decay) {
        count++;
        alpha += 1 * (reward > 0);
        beta += 1 * (reward <= 0);
    }

    // Upper Confident Bound
    float UCB(float T) {
        if (T == 0 || count == 0) return std::numeric_limits<float>::infinity();
        return std::sqrt(2 * std::log(T) / (float) count);
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

    size_t count = 0;

    float qi = 0; float ri = 0;

    void update(float reward, float decay) {
        count++;
        stddev *= decay;
        mu += (1 / (float) count) * (reward - mu);
        ri = reward;
        qi += std::pow(reward, 2);
    }

    float UCB(float T) {
        if (T <= 1 || count <= 1) return std::numeric_limits<float>::infinity();
        return std::sqrt(16 * (std::abs(qi - count * std::pow(ri, 2)) / (float) (count - 1)) * (std::log(T - 1) / (float) count));
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
