#include "bandit_arms.hpp"

namespace reilly {

namespace agents {

// Generic Bandit Arm

BanditArm::BanditArm() : count(0) {}

BanditArm::BanditArm(const BanditArm &other) : count(other.count) {}

BanditArm::~BanditArm() {}

// Bernoulli Arm

BernoulliArm::BernoulliArm(float alpha, float beta) : alpha(alpha), beta(beta) {}

BernoulliArm::BernoulliArm(const BernoulliArm &other) : BanditArm(other), alpha(other.alpha), beta(other.beta) {}

BernoulliArm &BernoulliArm::operator=(const BernoulliArm &other) {
    if (this != &other) {
        BernoulliArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.beta, beta);
    }
    return *this;
}

BernoulliArm::~BernoulliArm() {}

double BernoulliArm::operator()(std::minstd_rand &generator) {
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
    alpha += reward;
    beta += (1 - reward);
}

float BernoulliArm::UCB(float T) const {
    if (T == 0 || count == 0) return std::numeric_limits<float>::infinity();
    return std::sqrt(2 * std::log(T) / (float)count);
}

BernoulliArm::operator float() { return (float)alpha / (float)(alpha + beta); }

// Dynamic Bernuolli Arm

DynamicBernoulliArm::DynamicBernoulliArm(float alpha, float beta) : BernoulliArm(alpha, beta) {}

DynamicBernoulliArm::DynamicBernoulliArm(const DynamicBernoulliArm &other) : BernoulliArm(other) {}

DynamicBernoulliArm &DynamicBernoulliArm::operator=(const DynamicBernoulliArm &other) {
    if (this != &other) {
        DynamicBernoulliArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.beta, beta);
    }
    return *this;
}

DynamicBernoulliArm::~DynamicBernoulliArm() {}

void DynamicBernoulliArm::update(float reward, float gamma, float decay) {
    count++;
    alpha += reward;
    beta += (1 - reward);
    if (alpha + beta >= gamma) {
        alpha *= (gamma / (gamma + 1));
        beta *= (gamma / (gamma + 1));
    }
}

// Discounted Bernoulli Arm

DiscountedBernoulliArm::DiscountedBernoulliArm(float alpha, float beta) : BernoulliArm(alpha, beta), gamma(1), S(0), F(0) {}

DiscountedBernoulliArm::DiscountedBernoulliArm(const DiscountedBernoulliArm &other) : BernoulliArm(other), gamma(other.gamma), S(other.S), F(other.F) {}

DiscountedBernoulliArm &DiscountedBernoulliArm::operator=(const DiscountedBernoulliArm &other) {
    if (this != &other) {
        DiscountedBernoulliArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.beta, beta);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.S, S);
        std::swap(tmp.F, F);
    }
    return *this;
}

DiscountedBernoulliArm::~DiscountedBernoulliArm() {}

double DiscountedBernoulliArm::operator()(std::minstd_rand &generator) {
    S *= gamma; F *= gamma;  // Delayed discount
    std::gamma_distribution<> X(S + alpha, 1);
    std::gamma_distribution<> Y(F + beta, 1);
    float x = X(generator);
    float y = Y(generator);
    return x / (x + y);
}

void DiscountedBernoulliArm::update(float reward, float gamma, float decay) {
    count++;
    S += reward;
    F += (1 - reward);
    this->gamma = gamma;
}

DiscountedBernoulliArm::operator float() {
    S *= gamma; F *= gamma; // Delayed discount
    return (S + alpha) / ((S + alpha) + (F + beta));
}

// Gaussian Arm

GaussianArm::GaussianArm(float mu, float stddev) : ri(0), qi(0), mu(mu), stddev(stddev) {}

GaussianArm::GaussianArm(const GaussianArm &other) : BanditArm(other), ri(other.ri), qi(other.qi), mu(other.mu), stddev(other.stddev) {}

GaussianArm &GaussianArm::operator=(const GaussianArm &other) {
    if (this != &other) {
        GaussianArm tmp(other);
        std::swap(tmp.count, count);
        std::swap(tmp.mu, mu);
        std::swap(tmp.stddev, stddev);

        std::swap(tmp.ri, ri);
        std::swap(tmp.qi, qi);
    }
    return *this;
}

GaussianArm::~GaussianArm() {}

double GaussianArm::operator()(std::minstd_rand &generator) {
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

GaussianArm::operator float() { return mu; }

}  // namespace agents

}  // namespace reilly