import inspect
import functools
import aiolimiter
from typing import Awaitable, Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def rate_limited(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """
    Limits the rate at which the body of a function is entered
    using `aiolimiter.AsyncLimiter`.
    """
    signature = inspect.signature(func)

    rate_limiter_param_name = 'rate_limiter'

    if rate_limiter_param_name not in signature.parameters:
        raise TypeError(
            f"the function {func.__name__} was decorated with @rate_limited "
            f"but is missing the required '{rate_limiter_param_name}' parameter"
        )

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # extract rate limiter from parameters
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        limiter = bound.arguments.get(rate_limiter_param_name)

        if limiter is None:
            # if no limiter was provided, simply run the function
            return await func(*args, **kwargs)
        
        if not isinstance(limiter, aiolimiter.AsyncLimiter):
            raise TypeError("the rate limiter must be either None or an instance of aiolimiter.AsyncLimiter")
        
        async with limiter:
            return await func(*args, **kwargs)
        
    return wrapper