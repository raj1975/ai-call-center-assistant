import logging
import time
from functools import wraps
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_agent(agent_name: str) -> Callable:
    """Decorator that logs agent entry, exit, elapsed time, and any errors."""
    def decorator(fn: Callable) -> Callable:
        logger = get_logger(agent_name)

        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            logger.info("start | routing_in=%s", state.get("routing_decision", "—"))
            t0 = time.perf_counter()
            result = fn(state, *args, **kwargs)
            elapsed = time.perf_counter() - t0
            new_routing = result.get("routing_decision", "—") if isinstance(result, dict) else "—"
            new_errors = result.get("errors", []) if isinstance(result, dict) else []
            if new_errors:
                logger.warning("done  | %.2fs | routing_out=%s | errors=%s", elapsed, new_routing, new_errors)
            else:
                logger.info("done  | %.2fs | routing_out=%s", elapsed, new_routing)
            return result
        return wrapper
    return decorator
