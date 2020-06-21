#pragma once

#include "montecarlo.ipp"

namespace rl {

namespace agents {

class MonteCarloEveryVisit : public MonteCarlo {
   protected:
    void control();

   public:
    MonteCarloEveryVisit(size_t states, size_t actions, float epsilon, float gamma, float epsilon_decay = 1);
    MonteCarloEveryVisit(const MonteCarloEveryVisit &other);
    MonteCarloEveryVisit &operator=(const MonteCarloEveryVisit &other);
    ~MonteCarloEveryVisit();
};

}  // namespace agents

}  // namespace rl
