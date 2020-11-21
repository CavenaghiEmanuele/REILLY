#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <xtensor/xmath.hpp>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyvectorize.hpp>

#include <iostream>
#include <numeric>
#include <string>
#include <sstream>

#include <backend.hpp>
#include "virtual_overrides.ipp"

using namespace reilly::agents;

namespace py = pybind11;

PYBIND11_MODULE(backend, m) {
    m.doc() = "Reinforcement Learning Library Backend";

    py::class_<Agent, PyAgent>(m, "Agent")
        .def("get_action", &Agent::get_action)
        .def("reset", &Agent::reset, py::arg("init_state"))
        .def("update", &Agent::update,
            py::arg("next_state"),
            py::arg("reward"),
            py::arg("done")
        )
        .def("__repr__", &Agent::__repr__);
    
    py::class_<BernoulliArm>(m, "BernoulliArm")
        .def_readonly("alpha", &BernoulliArm::alpha)
        .def_readonly("beta", &BernoulliArm::beta)
        .def_readonly("count", &BernoulliArm::count);
    
    py::class_<MultiArmedBandit<BernoulliArm>, PyBernoulliBandit, Agent>(m, "BernoulliBandit");

    py::class_<GreedyBandit<BernoulliArm>, MultiArmedBandit<BernoulliArm>>(m, "BernoulliGreedyBandit", py::dynamic_attr())
        .def(py::init<size_t>(), py::arg("arms"))
        .def_readonly("arms", &GreedyBandit<BernoulliArm>::arms);
    
    py::class_<UCBBandit<BernoulliArm>, MultiArmedBandit<BernoulliArm>>(m, "BernoulliUCBBandit", py::dynamic_attr())
        .def(py::init<size_t>(), py::arg("arms"))
        .def_readonly("arms", &UCBBandit<BernoulliArm>::arms);
    
    py::class_<ThompsonSamplingBandit<BernoulliArm>, MultiArmedBandit<BernoulliArm>>(m, "BernoulliThompsonSamplingBandit", py::dynamic_attr())
        .def(py::init<size_t>(), py::arg("arms"))
        .def_readonly("arms", &ThompsonSamplingBandit<BernoulliArm>::arms);
    
    py::class_<DynamicBernoulliArm>(m, "DynamicBernoulliArm")
        .def_readonly("alpha", &DynamicBernoulliArm::alpha)
        .def_readonly("beta", &DynamicBernoulliArm::beta)
        .def_readonly("count", &DynamicBernoulliArm::count);
    
    py::class_<MultiArmedBandit<DynamicBernoulliArm>, PyDynamicBernoulliBandit, Agent>(m, "DynamicBernoulliBandit");
    
    py::class_<ThompsonSamplingBandit<DynamicBernoulliArm>, MultiArmedBandit<DynamicBernoulliArm>>(m, "DynamicBernoulliThompsonSamplingBandit", py::dynamic_attr())
        .def(py::init<size_t, float>(), py::arg("arms"), py::arg("gamma") = 1)
        .def_readonly("arms", &ThompsonSamplingBandit<DynamicBernoulliArm>::arms);
    
    py::class_<DiscountedBernoulliArm>(m, "DiscountedBernoulliArm")
        .def_readonly("alpha", &DiscountedBernoulliArm::alpha)
        .def_readonly("beta", &DiscountedBernoulliArm::beta)
        .def_readonly("count", &DiscountedBernoulliArm::count);
    
    py::class_<MultiArmedBandit<DiscountedBernoulliArm>, PyDiscountedBernoulliBandit, Agent>(m, "DiscountedBernoulliBandit");

    py::class_<ThompsonSamplingBandit<DiscountedBernoulliArm>, MultiArmedBandit<DiscountedBernoulliArm>>(m, "DiscountedBernoulliThompsonSamplingBandit")
        .def(py::init<size_t, float>(), py::arg("arms"), py::arg("gamma") = 1)
        .def_readonly("arms", &ThompsonSamplingBandit<DiscountedBernoulliArm>::arms);
    
    py::class_<GaussianArm>(m, "GaussianArm")
        .def_readonly("mu", &GaussianArm::mu)
        .def_readonly("stddev", &GaussianArm::stddev)
        .def_readonly("count", &GaussianArm::count);
    
    py::class_<MultiArmedBandit<GaussianArm>, PyGaussianBandit, Agent>(m, "GaussianBandit");

    py::class_<GreedyBandit<GaussianArm>, MultiArmedBandit<GaussianArm>>(m, "GaussianGreedyBandit", py::dynamic_attr())
        .def(py::init<size_t, float, float>(), py::arg("arms"), py::arg("gamma") = 1, py::arg("decay") = 1)
        .def_readonly("arms", &GreedyBandit<GaussianArm>::arms);
    
    py::class_<UCBBandit<GaussianArm>, MultiArmedBandit<GaussianArm>>(m, "GaussianUCBBandit", py::dynamic_attr())
        .def(py::init<size_t, float, float>(), py::arg("arms"), py::arg("gamma") = 1, py::arg("decay") = 1)
        .def_readonly("arms", &UCBBandit<GaussianArm>::arms);
    
    py::class_<ThompsonSamplingBandit<GaussianArm>, MultiArmedBandit<GaussianArm>>(m, "GaussianThompsonSamplingBandit", py::dynamic_attr())
        .def(py::init<size_t, float, float>(), py::arg("arms"), py::arg("gamma") = 1, py::arg("decay") = 1)
        .def_readonly("arms", &ThompsonSamplingBandit<GaussianArm>::arms);
    
    py::class_<TabularAgent, PyTabularAgent, Agent>(m, "TabularAgent");
    
    py::class_<MonteCarlo, PyMonteCarlo, TabularAgent>(m, "MonteCarlo");

    py::class_<MonteCarloFirstVisit, MonteCarlo>(m, "MonteCarloFirstVisit")
        .def(
            py::init<size_t, size_t, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<MonteCarloEveryVisit, MonteCarlo>(m, "MonteCarloEveryVisit")
        .def(
            py::init<size_t, size_t, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<TemporalDifference, PyTemporalDifference, TabularAgent>(m, "TemporalDifference");
    
    py::class_<Sarsa, TemporalDifference>(m, "Sarsa")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<QLearning, TemporalDifference>(m, "QLearning")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<ExpectedSarsa, TemporalDifference>(m, "ExpectedSarsa")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<DoubleTemporalDifference, PyDoubleTemporalDifference, TemporalDifference>(m, "DoubleTemporalDifference");

    py::class_<DoubleSarsa, DoubleTemporalDifference>(m, "DoubleSarsa")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<DoubleQLearning, DoubleTemporalDifference>(m, "DoubleQLearning")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<DoubleExpectedSarsa, DoubleTemporalDifference>(m, "DoubleExpectedSarsa")
        .def(
            py::init<size_t, size_t, float, float, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<NStep, PyNStep, TabularAgent>(m, "NStep");

    py::class_<NStepSarsa, NStep>(m, "NStepSarsa")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_step"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<NStepExpectedSarsa, NStep>(m, "NStepExpectedSarsa")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_step"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<NStepTreeBackup, NStep>(m, "NStepTreeBackup")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_step"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<QPlanning, PyQPlanning, TabularAgent>(m, "QPlanning");

    py::class_<TabularDynaQ, QPlanning>(m, "TabularDynaQ")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_plan"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<TabularDynaQPlus, QPlanning>(m, "TabularDynaQPlus")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_plan"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<PrioritizedSweeping, QPlanning>(m, "PrioritizedSweeping")
        .def(
            py::init<size_t, size_t, float, float, float, size_t, float, float>(),
            py::arg("states"),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_plan"),
            py::arg("theta"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<ApproximateAgent, PyApproximateAgent, Agent>(m, "ApproximateAgent")
        .def("reset", py::overload_cast<size_t>(&Agent::reset), py::arg("init_state"))
        .def("reset", py::overload_cast<std::vector<float>>(&ApproximateAgent::reset), py::arg("init_state"))
        .def("reset", py::overload_cast<py::array>(&ApproximateAgent::reset), py::arg("init_state"))
        .def("update", py::overload_cast<size_t, float, bool, py::kwargs>(&Agent::update),
            py::arg("next_state"),
            py::arg("reward"),
            py::arg("done")
        )
        .def("update", py::overload_cast<std::vector<float>, float, bool, py::kwargs>(&ApproximateAgent::update),
            py::arg("next_state"),
            py::arg("reward"),
            py::arg("done")
        )
        .def("update", py::overload_cast<py::array, float, bool, py::kwargs>(&ApproximateAgent::update),
            py::arg("next_state"),
            py::arg("reward"),
            py::arg("done")
        );
    
    py::class_<SemiGradientMonteCarlo, ApproximateAgent>(m, "SemiGradientMonteCarlo")
        .def(
            py::init<size_t, float, float, float, float, py::kwargs>(),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );

    py::class_<ApproximateTemporalDifference, PyApproximateTemporalDifference, ApproximateAgent>(m, "ApproximateTemporalDifference");

    py::class_<SemiGradientSarsa, ApproximateTemporalDifference>(m, "SemiGradientSarsa")
        .def(
            py::init<size_t, float, float, float, float, py::kwargs>(),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<SemiGradientExpectedSarsa, ApproximateTemporalDifference>(m, "SemiGradientExpectedSarsa")
        .def(
            py::init<size_t, float, float, float, float, py::kwargs>(),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<ApproximateNStep, PyApproximateNStep, ApproximateAgent>(m, "ApproximateNStep");

    py::class_<SemiGradientNStepSarsa, ApproximateNStep>(m, "SemiGradientNStepSarsa")
        .def(
            py::init<size_t, float, float, float, size_t, float, py::kwargs>(),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_step"),
            py::arg("epsilon_decay") = 1
        );
    
    py::class_<SemiGradientNStepExpectedSarsa, ApproximateNStep>(m, "SemiGradientNStepExpectedSarsa")
        .def(
            py::init<size_t, float, float, float, size_t, float, py::kwargs>(),
            py::arg("actions"),
            py::arg("alpha"),
            py::arg("epsilon"),
            py::arg("gamma"),
            py::arg("n_step"),
            py::arg("epsilon_decay") = 1
        );
}
