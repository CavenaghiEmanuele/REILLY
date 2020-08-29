#include "bandit_arms.hpp"

namespace reilly {

namespace agents {

// Generic Bandit Arm

BanditArm::BanditArm() : count(0), trace({0.5}) {}

BanditArm::BanditArm(const BanditArm &other) : count(other.count), trace(other.trace) {}

BanditArm::~BanditArm() {}

// Bernoulli Arm

BernoulliArm::BernoulliArm(float alpha, float beta) : alpha(alpha), beta(beta) {}

BernoulliArm::BernoulliArm(const BernoulliArm &other) : BanditArm(other), alpha(other.alpha), beta(other.beta) {}

BernoulliArm &BernoulliArm::operator=(const BernoulliArm &other) {
    if (this != &other) {
        BernoulliArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.trace, trace);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.beta, beta);
    }
    return *this;
}

BernoulliArm::~BernoulliArm() {}

double BernoulliArm::operator()(std::minstd_rand &generator) const {
    // X = Gamma(alpha, 1), Y = Gamma(beta, 1) ->
    // Z = X / X+Y = Beta(alpha, beta)
    std::gamma_distribution<> X(alpha, 1);
    std::gamma_distribution<> Y(beta, 1);
    float x = X(generator);
    float y = Y(generator);
    return x / (x + y);
}

void BernoulliArm::update(float reward, float gamma, float decay) {
    count++;
    alpha += 1 * (reward > 0);
    beta += 1 * (reward <= 0);
}

float BernoulliArm::UCB(float T) const {
    if (T == 0 || count == 0) return std::numeric_limits<float>::infinity();
    return std::sqrt(2 * std::log(T) / (float)count);
}

BernoulliArm::operator float() const { return (float)alpha / (float)(alpha + beta); }

// Gaussian Arm

GaussianArm::GaussianArm(float mu, float stddev) : ri(0), qi(0), mu(mu), stddev(stddev) {}

GaussianArm::GaussianArm(const GaussianArm &other) : BanditArm(other), ri(other.ri), qi(other.qi), mu(other.mu), stddev(other.stddev) {}

GaussianArm &GaussianArm::operator=(const GaussianArm &other) {
    if (this != &other) {
        GaussianArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.trace, trace);
        std::swap(tmp.mu, mu);
        std::swap(tmp.stddev, stddev);

        // std::swap(tmp.ri, ri);
        // std::swap(tmp.qi, qi);
    }
    return *this;
}

GaussianArm::~GaussianArm() {}

double GaussianArm::operator()(std::minstd_rand &generator) const {
    std::normal_distribution<> normal(mu, stddev);
    return normal(generator);
}

void GaussianArm::update(float reward, float gamma, float decay) {
    count++;
    stddev *= decay;
    mu += (1 / (float)count) * (reward - mu);

    ri = reward; qi += std::pow(reward, 2);
}

float GaussianArm::UCB(float T) const {
    if (T <= 1 || count <= 1) return std::numeric_limits<float>::infinity();
    return std::sqrt(16 * (std::abs(qi - count * std::pow(ri, 2)) / (float)(count - 1)) * (std::log(T - 1) / (float)count));
}

GaussianArm::operator float() const { return mu; }

}  // namespace agents

}  // namespace reilly