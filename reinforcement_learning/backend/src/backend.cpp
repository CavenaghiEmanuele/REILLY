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

using namespace rl::agents;

namespace py = pybind11;

PYBIND11_MODULE(backend, m) {
    m.doc() = "Reinforcement Learning Library Backend";

    py::class_<Agent, PyAgent>(m, "Agent")
        .def("get_action", &Agent::get_action)
        .def("reset", &Agent::reset, py::arg("init_state"))
        .def("update", &Agent::update,
            py::arg("next_state"),
            py::arg("reward"),
            py::arg("done"),
            py::arg("training")
        )
        .def("__repr__", &Agent::__repr__);
    
    py::class_<MonteCarlo, PyMonteCarlo, Agent>(m, "MonteCarlo");

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
    
    py::class_<TemporalDifference, PyTemporalDifference, Agent>(m, "TemporalDifference");
    
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
}
