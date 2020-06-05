from .agents import (
    MonteCarloAgent, 
    TemporalDifference, SarsaAgent, QLearningAgent, ExpectedSarsaAgent,
    DoubleTemporalDifference, DoubleSarsaAgent, DoubleQLearningAgent, DoubleExpectedSarsaAgent,
    NStep, NStepSarsaAgent, NStepExpectedSarsaAgent
    )
from .environments import Environment, Taxi, Frozen_Lake4x4, Frozen_Lake8x8, TextEnvironment, TextNeighbor
from .sessions import Session
from .structures import StateValue, ActionValue, Policy
from .utils import plot
