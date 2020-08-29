#pragma once

#include <pybind11/pybind11.h>

#include <backend.hpp>

namespace reilly {

namespace agents {

class PyAgent : public Agent {
   public:
    using Agent::Agent;
    
    size_t get_action() override { PYBIND11_OVERLOAD(size_t, Agent, get_action, ); }
    void reset(size_t init_state) override { PYBIND11_OVERLOAD_PURE(void, Agent, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD_PURE(std::string, Agent, __repr__, );
    }
};

class PyBernoulliBandit : public MultiArmedBandit<BernoulliArm> {
    public:
     using MultiArmedBandit<BernoulliArm>::MultiArmedBandit;

    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, MultiArmedBandit, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyDynamicBernoulliBandit : public MultiArmedBandit<DynamicBernoulliArm> {
    public:
     using MultiArmedBandit<DynamicBernoulliArm>::MultiArmedBandit;

    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, MultiArmedBandit, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyDiscountedBernoulliBandit : public MultiArmedBandit<DiscountedBernoulliArm> {
    public:
     using MultiArmedBandit<DiscountedBernoulliArm>::MultiArmedBandit;

    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, MultiArmedBandit, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyGaussianBandit : public MultiArmedBandit<GaussianArm> {
    public:
     using MultiArmedBandit<GaussianArm>::MultiArmedBandit;

    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, MultiArmedBandit, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyTabularAgent : public TabularAgent {
   public:
    using TabularAgent::TabularAgent;
    
    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, TabularAgent, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD(void, Agent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyMonteCarlo : public MonteCarlo {
   public:
    using MonteCarlo::MonteCarlo;

    void control() override { PYBIND11_OVERLOAD_PURE(void, MonteCarlo, control, ); }
};

class PyTemporalDifference : public TemporalDifference {
   public:
    using TemporalDifference::TemporalDifference;

    void reset(size_t init_state) override { PYBIND11_OVERLOAD(void, TemporalDifference, reset, init_state); }
    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

class PyDoubleTemporalDifference : public DoubleTemporalDifference {
   public:
    using DoubleTemporalDifference::DoubleTemporalDifference;

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

class PyNStep : public NStep {
   public:
    using NStep::NStep;

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

class PyQPlanning : public QPlanning {
   public:
    using QPlanning::QPlanning;

    void update(size_t next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

class PyApproximateAgent : public ApproximateAgent {
   public:
    using ApproximateAgent::ApproximateAgent;
    
    void reset(Vector init_state) override { PYBIND11_OVERLOAD_PURE(void, ApproximateAgent, reset, init_state); }
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, ApproximateAgent, update, next_state, reward, done, kwargs);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyApproximateTemporalDifference : public ApproximateTemporalDifference {
   public:
    using ApproximateTemporalDifference::ApproximateTemporalDifference;

    void reset(Vector init_state) override { PYBIND11_OVERLOAD(void, ApproximateTemporalDifference, reset, init_state); }
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

class PyApproximateNStep : public ApproximateNStep {
   public:
    using ApproximateNStep::ApproximateNStep;

    void reset(Vector init_state) override { PYBIND11_OVERLOAD(void, ApproximateNStep, reset, init_state); }
    void update(Vector next_state, float reward, bool done, py::kwargs kwargs) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, kwargs);
    }
};

}  // namespace agents

}  // namespace reilly
