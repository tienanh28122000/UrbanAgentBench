from enum import Enum

from urban_agent_bench.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from urban_agent_bench.data_model.tasks import RewardType, Task
from urban_agent_bench.evaluator.evaluator_action import ActionEvaluator
from urban_agent_bench.evaluator.evaluator_communicate import CommunicateEvaluator
from urban_agent_bench.evaluator.evaluator_env import EnvironmentEvaluator
from urban_agent_bench.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from urban_agent_bench.registry import registry


class EvaluationType(str, Enum):
    ENV = "env"
    COMMUNICATE = "communicate"
    ACTION = "action"
    ACTION_NL = "action_nl"
    ALL = "all"
    NL_ASSERTIONS = "nl_assertions"  # WIP
    ALL_WITH_NL_ASSERTIONS = "all_with_nl_assertions"  # WIP


def _merge_reward_breakdowns(*reward_infos: RewardInfo) -> dict[RewardType, float]:
    reward_breakdown = {}
    for reward_info in reward_infos:
        if reward_info.reward_breakdown is not None:
            reward_breakdown.update(reward_info.reward_breakdown)
    return reward_breakdown


def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    solo_mode: bool,
    domain: str,
) -> RewardInfo:
    """
    Evaluate the simulation based on the evaluation type.
    """
    if simulation.termination_reason not in {
        TerminationReason.AGENT_STOP,
        TerminationReason.USER_STOP,
    }:
        return RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={
                "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason.value}"
            },
        )
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            reward_basis=None,
            info={"note": "No evaluation criteria"},
        )
    if evaluation_type == EvaluationType.ENV:
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor(domain),
            task=task,
            full_trajectory=simulation.messages,
            solo_mode=solo_mode,
        )
    elif evaluation_type == EvaluationType.NL_ASSERTIONS:
        reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type == EvaluationType.COMMUNICATE:
        reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type == EvaluationType.ACTION:
        reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
    elif evaluation_type == EvaluationType.ACTION_NL:
        action_reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
        nl_reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )

        reward = action_reward_info.reward * nl_reward_info.reward
        reward_breakdown = _merge_reward_breakdowns(
            action_reward_info, nl_reward_info
        )

        reward_info = RewardInfo(
            reward=reward,
            action_checks=action_reward_info.action_checks,
            nl_assertions=nl_reward_info.nl_assertions,
            reward_basis=[RewardType.ACTION, RewardType.NL_ASSERTION],
            reward_breakdown=reward_breakdown,
            info={
                "action": action_reward_info.info,
                "nl": nl_reward_info.info,
            },
        )
    elif evaluation_type in {EvaluationType.ALL, EvaluationType.ALL_WITH_NL_ASSERTIONS}:
        env_reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor(domain),
            task=task,
            full_trajectory=simulation.messages,
            solo_mode=solo_mode,
        )
        action_reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
        communicate_reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
        )
        nl_reward_info = None
        if evaluation_type == EvaluationType.ALL_WITH_NL_ASSERTIONS:
            nl_reward_info = NLAssertionsEvaluator.calculate_reward(
                task=task,
                full_trajectory=simulation.messages,
            )

        reward_infos = [env_reward_info, action_reward_info, communicate_reward_info]
        reward_basis = [
            RewardType.DB,
            RewardType.ENV_ASSERTION,
            RewardType.ACTION,
            RewardType.COMMUNICATE,
        ]
        if nl_reward_info is not None:
            reward_infos.append(nl_reward_info)
            reward_basis.append(RewardType.NL_ASSERTION)

        reward = 1.0
        for reward_component in reward_infos:
            reward *= reward_component.reward
        reward_breakdown = _merge_reward_breakdowns(*reward_infos)

        reward_info = RewardInfo(
            reward=reward,
            db_check=env_reward_info.db_check,
            env_assertions=env_reward_info.env_assertions,
            action_checks=action_reward_info.action_checks,
            nl_assertions=nl_reward_info.nl_assertions
            if nl_reward_info is not None
            else None,
            communicate_checks=communicate_reward_info.communicate_checks,
            reward_basis=reward_basis,
            reward_breakdown=reward_breakdown,
            info={
                "env": env_reward_info.info,
                "nl": nl_reward_info.info if nl_reward_info is not None else None,
                "communicate": communicate_reward_info.info,
                "action": action_reward_info.info,
            },
        )
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    return reward_info
