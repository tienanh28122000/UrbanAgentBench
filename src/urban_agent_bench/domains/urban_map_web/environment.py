from pathlib import Path
from typing import Optional

from urban_agent_bench.data_model.tasks import Task
from urban_agent_bench.domains.urban_map_web.data_model import UrbanWebDB
from urban_agent_bench.domains.urban_map_web.tools import UrbanWebTools
from urban_agent_bench.domains.urban_map_web.utils import (
    URBAN_WEB_DB_PATH,
    URBAN_WEB_POLICY_PATH,
    URBAN_WEB_TASK_SET_PATH,
)
from urban_agent_bench.environment.environment import Environment
from urban_agent_bench.utils import load_file

def get_environment(
    db: Optional[UrbanWebDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """Initializes the completely isolated Urban Web Environment."""
    if solo_mode:
        raise ValueError("Urban Map Web domain does not support solo mode.")
    
    # Load the static snapshot database
    if db is None:
        db = UrbanWebDB.load(URBAN_WEB_DB_PATH)
        
    tools = UrbanWebTools(db)
    
    with open(URBAN_WEB_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
        
    return Environment(
        domain_name="urban_map_web",
        policy=policy,
        tools=tools,
    )

def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Loads benchmark tasks from the predefined set."""
    tasks = load_file(URBAN_WEB_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    
    if task_split_name is None:
        return tasks
        
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(f"Invalid task split name. Valid splits are: {list(task_splits.keys())}")
        
    tasks = [task for task in tasks if task.id in task_splits[task_split_name]]
    return tasks

def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(URBAN_WEB_TASK_SET_PATH).parent
        / f"split_{Path(URBAN_WEB_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)