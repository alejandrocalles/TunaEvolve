import logging
import asyncio
import dataclasses
from typing import Dict, List, Union, Optional

import aiolimiter
from pydantic import BaseModel

from .query import sample_model_kwargs, query_async
from .models import QueryResult
from .dynamic_sampling import BanditBase, FixedSampler

MAX_RETRIES = 3

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AsyncClientConfig:
    """Configuration specific to the AsyncLLMClient.
    
    Attributes:
        max_parallel_queries:
            The maximum number of queries that should be able
            to run in parallel.
            Its length should match the number of model names provided,
            and the limit will apply on a per-model basis.

        min_seconds_between_api_requests:
            The rate at which queries should be dispatched.
            Its length should match the number of model names provided. 
            This rate will apply on a per-model basis.
    """
    max_parallel_queries: List[int]
    min_seconds_between_api_requests: List[float]


class AsyncLLMClient:
    """
    An LLM Client that supports asynchronous queries.
    """

    def __init__(
        self,
        config: AsyncClientConfig,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[BaseModel] = None,
        verbose: bool = True,
    ):
        """Initialize this client.

        Args:
            config:
                Configuration specific to the AsyncLLMClient.
            model_names:
                A model name or a list of model names. Each
                name must be compatible with the function `get_llm_client`
                in `shinka/llm/client.py`.
            temperatures:
                The temperature(s) for generation. If a list,
                its length should match the number of model names provided.
            max_tokens:
                The maximum number of tokens. If a list,
                its length should match the number of models names provided.
        """
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None

        if len(config.max_parallel_queries) != len(model_names):
            raise ValueError("the length of max_parallel_queries should match the number of model names provided")
        # map each model name to its query scheduler
        self.query_schedulers = {
            model_name: asyncio.Semaphore(max_queries)
            for model_name, max_queries in zip(model_names, config.max_parallel_queries)
        }

        if len(config.min_seconds_between_api_requests) != len(model_names):
            raise ValueError("the length of min_seconds_between_api_requests should match the number of model names provided")
        # map each model name to its rate limiter
        self.rate_limiters = {
            model_name: aiolimiter.AsyncLimiter(max_rate=1, time_period=period)
            for model_name, period in zip(model_names, config.min_seconds_between_api_requests)
        }

        self.verbose = verbose

    async def batch_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: List[Dict] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
        # Repeat msg, system_msg, msg_history num_samples times
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

        # Create async tasks
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    llm_kwargs[i],
                    num_samples,
                )
            )

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and filter out exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

        # Print batch total cost
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    async def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """Batch query the LLM with the given message and system message asynchronously.

        Args:
            msg (str): The message to query the LLM with.
            system_msg (str): The system message to query the LLM with.
        """
        # Repeat msg, system_msg, msg_history num_samples times
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

        # Get posterior probabilities
        posterior = self.llm_selection.posterior(samples=num_samples)
        if self.verbose:
            lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))

        # Create async tasks
        tasks = []
        for i in range(len(msg)):
            tasks.append(
                self._sample_kwargs_query_async_with_retry(
                    i,
                    msg[i],
                    system_msg[i],
                    msg_history[i],
                    posterior,
                    num_samples,
                )
            )

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and filter out exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch query task {i}: {str(result)}")
            elif result is not None and len(result) > 1 and result[1] is not None:
                final_results.append(result[1])

        # Print batch total cost
        if self.verbose:
            total_cost = sum(
                r.cost
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            )
            formatted_costs = [
                f"{r.cost:.4f}"
                for r in final_results
                if hasattr(r, "cost") and r.cost is not None
            ]
            logger.info(f"==> SAMPLING: Individual API costs: {formatted_costs}")
            logger.info(f"==> SAMPLING: Total API costs: ${total_cost:.4f}")
        return final_results

    def get_kwargs(self):
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    async def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """Execute a single query to the LLM asynchronously.

        Args:
            msg: The message to query the LLM with.
            system_msg: The system message to query the LLM with.
            msg_history: Message history. Defaults to [].
            llm_kwargs: Additional LLM parameters. Defaults to {}.

        Returns:
            The result of the query.
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

        # Get posterior probabilities and create model_posteriors dict
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        model_name = llm_kwargs["model_name"]
        async with self.query_schedulers[model_name]: # limits the number of parallel queries
            for try_count in range(MAX_RETRIES):
                try:
                    result = await query_async(
                        msg=msg,
                        system_msg=system_msg,
                        msg_history=msg_history,
                        output_model=self.output_model,
                        model_posteriors=model_posteriors,
                        rate_limiter=self.rate_limiters[model_name],
                        **llm_kwargs,
                    )
                    if self.verbose and hasattr(result, "cost") and result.cost is not None:
                        logger.info(f"==> QUERY: API cost: ${result.cost:.4f}")
                    return result
                except Exception as e:
                    logger.error(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                    await asyncio.sleep(1)  # Add delay between retries
        return None

    async def _query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        kwargs: Dict = {},
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        model_name = kwargs["model_name"] # the model name is not optional anyways
        async with self.query_schedulers[model_name]:
            for try_count in range(MAX_RETRIES):
                try:
                    result = await query_async(
                        msg=msg,
                        system_msg=system_msg,
                        msg_history=msg_history,
                        output_model=self.output_model,
                        rate_limiter=self.rate_limiters[model_name],
                        **kwargs,
                    )
                    return idx, result
                except Exception as e:
                    logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                    await asyncio.sleep(1)  # Add delay between retries
        return idx, None

    async def _sample_kwargs_query_async_with_retry(
        self,
        idx: int,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        model_sample_probs: Optional[List[float]] = None,
        total_samples: int = 1,
    ) -> tuple[int, Optional[QueryResult]]:
        kwargs = sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

        # Create model_posteriors dict from model_names and model_sample_probs
        model_posteriors = None
        if model_sample_probs is not None and isinstance(self.model_names, list):
            model_posteriors = dict(zip(self.model_names, model_sample_probs))
            model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        if self.verbose:
            logger.info(
                f"==> SAMPLING: {idx + 1}/{total_samples} {list(kwargs.values())}"
            )

        model_name = kwargs["model_name"]
        async with self.query_schedulers[model_name]:
            for try_count in range(MAX_RETRIES):
                try:
                    result = await query_async(
                        msg=msg,
                        system_msg=system_msg,
                        msg_history=msg_history,
                        output_model=self.output_model,
                        model_posteriors=model_posteriors,
                        rate_limiter=self.rate_limiters[model_name],
                        **kwargs,
                    )
                    return idx, result
                except Exception as e:
                    logger.info(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                    await asyncio.sleep(1)  # Add delay between retries
        return idx, None

