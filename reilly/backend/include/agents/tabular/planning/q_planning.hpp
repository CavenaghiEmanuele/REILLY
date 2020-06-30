#pragma once

#include "../tabular_agent.ipp"

namespace rl {

namespace agents {

class QPlanning : public TabularAgent {
   protected:
    struct Result {
        float reward;
        size_t next_state;
    };

    class Model {
       private:
        xt::xtensor<int64_t, 2> visited;
        std::vector<Result> results;

       public:
        Model(size_t states, size_t actions) {visited = -(xt::ones<int64_t>({states, actions})); }
        Model(const Model &other) : visited(other.visited), results(other.results) {}
        Model &operator==(const Model &other) {
            if (this != &other) {
                Model tmp(other);
                std::swap(tmp.visited, visited);
                std::swap(tmp.results, results);
            }
            return *this;
        }
        ~Model() {}

        Result operator()(size_t state, size_t action) {
            Result out = {0, state};
            int64_t index = visited(state, action);
            if (index >= 0) out = results[index];
            return out;
        }

        void set_result(size_t state, size_t action, float reward, size_t next_state) {
            int64_t index = visited(state, action);
            if (index >= 0) {
                results[index].reward = reward;
                results[index].next_state = next_state;
            } else {
                results.push_back({reward, next_state});
                visited(state, action) = results.size() - 1;
            }
        }

        size_t get_random_observed_state() {
            auto idx = xt::from_indices(xt::where(xt::greater_equal(visited, 0)));
            return xt::random::choice(xt::row(idx, 0), 1)(0);
        }

        size_t get_random_observed_action(size_t observed_state) {
            auto idx = xt::from_indices(xt::where(xt::greater_equal(xt::row(visited, observed_state), 0)));
            return xt::random::choice(xt::row(idx, 0), 1)(0);
        }

        std::vector<Point> lead_to(size_t state) {
            std::vector<Point> leaders;
            auto idx = xt::argwhere(xt::greater_equal(visited, 0));
            for (auto i : idx) {
                Result r = results[visited(i[0], i[1])];
                if (r.next_state == state) leaders.push_back({i[0], i[1], r.reward});
            }
            return leaders;
        }
    };

    size_t n_plan;
    Model model;

   public:
    QPlanning(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float epsilon_decay = 1);
    QPlanning(const QPlanning &other);
    virtual ~QPlanning();

    void reset(size_t init_state);

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl