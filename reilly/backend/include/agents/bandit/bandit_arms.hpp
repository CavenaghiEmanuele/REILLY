#include "../agent.ipp"

namespace reilly {

namespace agents {

class BanditArm {
   public:
    size_t count;

    BanditArm();
    BanditArm(const BanditArm &other);
    virtual ~BanditArm();

    virtual double operator()(std::minstd_rand &generator) = 0;
    virtual void update(float reward, float gamma, float decay) = 0;
    virtual float UCB(float T) const = 0;  // Upper Confident Bound
    virtual operator float() = 0;
};

class BernoulliArm : public BanditArm {
   public:
    float alpha;
    float beta;

    BernoulliArm(float alpha = 1, float beta = 1);
    BernoulliArm(const BernoulliArm &other);
    BernoulliArm &operator=(const BernoulliArm &other);
    ~BernoulliArm();

    virtual double operator()(std::minstd_rand &generator);
    virtual void update(float reward, float gamma, float decay);
    virtual float UCB(float T) const;
    virtual operator float();
};

class DynamicBernoulliArm : public BernoulliArm {
   public:
    DynamicBernoulliArm(float alpha = 2, float beta = 2);
    DynamicBernoulliArm(const DynamicBernoulliArm &other);
    DynamicBernoulliArm &operator=(const DynamicBernoulliArm &other);
    ~DynamicBernoulliArm();

    virtual void update(float reward, float gamma, float decay);
};

class DiscountedBernoulliArm : public BernoulliArm {
   private:
    float gamma;
    float S;
    float F;

   public:
    DiscountedBernoulliArm(float alpha = 1, float beta = 1);
    DiscountedBernoulliArm(const DiscountedBernoulliArm &other);
    DiscountedBernoulliArm &operator=(const DiscountedBernoulliArm &other);
    ~DiscountedBernoulliArm();

    virtual double operator()(std::minstd_rand &generator);
    virtual void update(float reward, float gamma, float decay);
    virtual operator float();
};

class GaussianArm : public BanditArm {
   private:
    float ri;
    float qi;

   public:
    float mu;
    float stddev;

    GaussianArm(float mu = 0.5, float stddev = 1);
    GaussianArm(const GaussianArm &other);
    GaussianArm &operator=(const GaussianArm &other);
    ~GaussianArm();

    virtual double operator()(std::minstd_rand &generator);
    virtual void update(float reward, float gamma, float decay);
    virtual float UCB(float T) const;
    virtual operator float();
};

}  // namespace agents

}  // namespace reilly