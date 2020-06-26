#pragma once

#include "prioritized_sweeping.hpp"

namespace rl {

namespace agents {

PrioritizedSweeping::PrioritizedSweeping(size_t states, size_t actions, float alpha, float epsilon, float gamma, size_t n_plan, float theta, float epsilon_decay)
    : QPlanning(states, actions, alpha, epsilon, gamma, n_plan, epsilon_decay), theta(theta) {}

PrioritizedSweeping::PrioritizedSweeping(const PrioritizedSweeping &other) : QPlanning(other), theta(other.theta) {}

PrioritizedSweeping &PrioritizedSweeping::operator==(const PrioritizedSweeping &other) {
    if (this != &other) {
        PrioritizedSweeping tmp(other);
        std::swap(tmp.states, states);
        std::swap(tmp.actions, actions);
        std::swap(tmp.Q, Q);
        std::swap(tmp.pi, pi);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
        std::swap(tmp.n_plan, n_plan);
        std::swap(tmp.theta, theta);
        std::swap(tmp.model, model);
        std::swap(tmp.pqueue, pqueue);
    }
    return *this;
}

PrioritizedSweeping::~PrioritizedSweeping() {}

void PrioritizedSweeping::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        model.set_result(state, action, reward, next_state);
        float priority = std::abs(reward + gamma * xt::amax(xt::row(Q, next_state))() - Q(state, action));
        if (priority > theta) pqueue.push({state, action, priority});
        for (size_t i = 0; i < n_plan && !pqueue.empty(); i++) {
            Point p = pqueue.top(); pqueue.pop();
            Result r = model(p.state, p.action);
            Q(p.state, p.action) += alpha * (r.reward + gamma * xt::amax(xt::row(Q, r.next_state))() - Q(p.state, p.action));
            policy_update(Q, pi, p.state);
            std::vector<Point> leaders = model.lead_to(p.state);
            for (Point l : leaders) {
                priority = std::abs(l.reward + gamma * xt::amax(xt::row(Q, p.state))() - Q(l.state, l.action));
                if (priority > theta) pqueue.push({l.state, l.action, priority});
            }
        }
        while (!pqueue.empty()) pqueue.pop();    // We need to empty the queue
    }

    state = next_state;
    action = select_action(pi, next_state);

    if (done) epsilon *= epsilon_decay;
}

std::string PrioritizedSweeping::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon << ", gamma=" << gamma ;
    out << ", n_plan=" << n_plan << ", theta=" << theta << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl
