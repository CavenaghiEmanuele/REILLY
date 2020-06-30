#pragma once

#include <agents/agent.ipp>

#include <agents/tabular/tabular_agent.ipp>
#include <agents/tabular/montecarlo/montecarlo.ipp>
#include <agents/tabular/montecarlo/first_visit.ipp>
#include <agents/tabular/montecarlo/every_visit.ipp>
#include <agents/tabular/temporal_difference/temporal_difference.ipp>
#include <agents/tabular/temporal_difference/sarsa.ipp>
#include <agents/tabular/temporal_difference/q_learning.ipp>
#include <agents/tabular/temporal_difference/expected_sarsa.ipp>
#include <agents/tabular/temporal_difference/double_learning/double_temporal_difference.ipp>
#include <agents/tabular/temporal_difference/double_learning/double_sarsa.ipp>
#include <agents/tabular/temporal_difference/double_learning/double_q_learning.ipp>
#include <agents/tabular/temporal_difference/double_learning/double_expected_sarsa.ipp>
#include <agents/tabular/n_step/n_step.ipp>
#include <agents/tabular/n_step/n_step_sarsa.ipp>
#include <agents/tabular/n_step/n_step_expected_sarsa.ipp>
#include <agents/tabular/n_step/n_step_tree_backup.ipp>
#include <agents/tabular/planning/q_planning.ipp>
#include <agents/tabular/planning/tabular_dyna_q.ipp>
#include <agents/tabular/planning/tabular_dyna_q_plus.ipp>
#include <agents/tabular/planning/prioritized_sweeping.ipp>

#include <agents/approximate/approximate_agent.ipp>
#include <agents/approximate/temporal_difference/approximate_temporal_difference.ipp>
#include <agents/approximate/temporal_difference/semigradient_sarsa.ipp>
#include <agents/approximate/temporal_difference/semigradient_expected_sarsa.ipp>
