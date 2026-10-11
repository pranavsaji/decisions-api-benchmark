"""The three benchmark tasks, keyed by name."""
from bench.tasks import injection, intent, relevance

TASKS = {task.name: task for task in (intent.TASK, injection.TASK, relevance.TASK)}
