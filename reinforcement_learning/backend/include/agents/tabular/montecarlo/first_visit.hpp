#pragma once

#include "montecarlo.ipp"

namespace rl {

namespace agents {

class MonteCarloFirstVisit : public MonteCarlo {
   protected:
    void control();

   public:
    MonteCarloFirstVisit(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay = 1);
    MonteCarloFirstVisit(const MonteCarloFirstVisit &other);
    MonteCarloFirstVisit &operator=(const MonteCarloFirstVisit &other);
    ~MonteCarloFirstVisit();
};

}  // namespace agents

}  // namespace rl
