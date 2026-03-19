"""Datasets for RL Training of LLMs."""
from shinka import database
import logging
from typing import List, Dict, Tuple, Optional, Union
import math
import random
import itertools
import datasets
import dataclasses
import pathlib

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Message:
    """Represents a message in a conversation with an LLM."""
    role: str
    content:str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

def build_dpo_dataset(
    programs: List[database.Program],
    save_path: Optional[pathlib.Path] = None,
) -> Union[datasets.Dataset, datasets.IterableDataset]:
    """Returns a preference dataset from the given list of programs for DPO training.

    See https://huggingface.co/docs/trl/en/dataset_formats#preference for more
    information on this dataset format.

    Args:
        programs:
            A list of programs to build the dataset from.

            All programs should be compatible with the
            `extract_prompt_and_response()` function below.

            It is expected that every generation of programs will have
            multiple correct programs per generation and that all the programs
            in a given generation will have have the same
            prompt as extracted by `extract_prompt_and_response()`.
        save_path:
            If not `None` the generated dataset will be saved to this path.
    """

    generations: Dict[int, List[database.Program]] = {}
    for program in programs:
        generations.setdefault(program.generation, []).append(program)
    
    logger.info(f"Generating preference dataset, {len(generations)} generation(s) found in database")

    def row_generator():
        """Yields rows for the DPO dataset"""
        # we group the programs by generation because we assume all programs
        # within the same generation have the same prompt
        # thus, we will only pair-up programs from the same generation

        for gen_id, generation in generations.items():
            # separate the programs by correctness 
            correct_programs = [p for p in generation if p.correct]
            incorrect_programs = [p for p in generation if not p.correct]

            logger.info(
                f"Gen {gen_id} has {len(correct_programs)} correct program(s) "
                f"and {len(incorrect_programs)} incorrect program(s)"
            )

            # cluster the correct programs by score
            clusters: dict[float, List[database.Program]] = {}
            for program in correct_programs:
                clusters.setdefault(program.combined_score, []).append(program)
            
            unique_scores = sorted(clusters.keys())
            if len(unique_scores) < 2:
                logger.warning(
                    f"Skipping Gen {gen_id} for DPO dataset: "
                    f"found {len(unique_scores)} unique scores among {len(generation)} programs"
                )
                continue # we need at least two unique scores to form pairs
            
            logger.info(
                f"Gen {gen_id} has {len(unique_scores)} unique scores, "
                "starting creation of preference pairs"
            )

            def preference_pairs_generator():
                """Yield preference pairs of programs as a tuple (rejected, chosen)"""
                # separate low scores from high scores
                mid = math.ceil(len(unique_scores) / 2)
                low_scores, high_scores = unique_scores[:mid], unique_scores[mid:]
                for low_score, high_score in zip(low_scores, high_scores):
                    rejected_program = random.choice(clusters[low_score])
                    chosen_program = random.choice(clusters[high_score])
                    # preference pair of (low_score, high_score)
                    yield rejected_program, chosen_program
                if not correct_programs:
                    return
                for incorrect_program, correct_program in zip(incorrect_programs, itertools.cycle(correct_programs)):
                    # preference pair of (incorrect, correct)
                    yield incorrect_program, correct_program
            
            for rejected_program, chosen_program in preference_pairs_generator():
                if (result_rejected := extract_prompt_and_response(rejected_program)) is None:
                    continue
                rejected_prompt, rejected_response = result_rejected

                if (result_chosen := extract_prompt_and_response(chosen_program)) is None:
                    continue
                chosen_prompt, chosen_response = result_chosen

                if chosen_prompt != rejected_prompt:
                    logger.error(
                        f"Prompt mismatch in Gen {gen_id}: "
                        f"Branch {chosen_program.branch_id} (Chosen) "
                        f"and Branch {rejected_program.branch_id} (Rejected) "
                        f"had different initial prompts"
                    )
                    raise ValueError("Expected programs in the same generation to have the same initial prompt, found two differnt prompts.")
                
                # for a dataset in conversational format,
                # all three columns ('prompt', 'chosen', 'rejected')
                # expect a list of messages
                # so we create lists of length 1
                yield dict(
                    generation=gen_id,
                    prompt=[chosen_prompt.to_dict()],
                    chosen=[chosen_response.to_dict()],
                    rejected=[rejected_response.to_dict()],
                    chosen_branch_id=chosen_program.branch_id,
                    rejected_branch_id=rejected_program.branch_id,
                )

    dataset_schema = datasets.Features({
            "generation": datasets.Value('int64'),
            "prompt": datasets.List({"role": datasets.Value('string'), "content": datasets.Value('string')}),
            "chosen": datasets.List({"role": datasets.Value('string'), "content": datasets.Value('string')}),
            "rejected": datasets.List({"role": datasets.Value('string'), "content": datasets.Value('string')}),
            "chosen_branch_id": datasets.Value('int64'),
            "rejected_branch_id": datasets.Value('int64'),
    })

    # the row_generator is meant to be used with Dataset.from_generator
    # for efficieny, however, for debugging purposes, we materialize
    # the rows here and then use Dataset.from_list instead
    rows = [row for row in row_generator()]

    if not rows:
        message = "DPO PREFERENCE DATASET NOT GENERATED because no generation had enough valid programs"
        logger.fatal(message)
        raise ValueError(message)
    
    try:
        dataset = datasets.Dataset.from_list(
            rows,
            features=dataset_schema,
        )
    except Exception as e:
        logger.fatal(f"DPO PREFERENCE DATASET NOT GENERATED due to an exception")
        raise e

    logger.info(f"DPO PREFERENCE DATASET GENERATED")

    if not isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)):
        raise TypeError("DPO dataset must be an instance of datasets.Dataset or datasets.IterableDataset")
    
    if save_path is not None:
        if isinstance(dataset, datasets.Dataset):
            logger.info(f"Saving preference dataset to {save_path}")
            dataset.save_to_disk(save_path)
        else:
            logger.warning(f"Unable to save preference dataset")

    return dataset

def extract_prompt_and_response(program: database.Program) -> Optional[Tuple[Message, Message]]:
    """
    Extracts the original prompt and the last response from the conversation that
    generated a program.
    
    Args:
        program:
            The program from which to extract the prompt and response.
            This information is fetched from the program's `metadata` field,
            which should contain the `llm_result` key. Setting this field
            is the responsibility of the runner.
    
    Returns:
        If the information is succesfully fetched, returns a tuple (prompt, response),
        where prompt is a dict with the first message in the conversation and response
        is a dict with the last message in the conversation.
        If the last message does not have the correct role or if the
        `llm_result` field is not present, returns None.
    
    Raises:
        KeyError: The relevant messages did not have the correct keys.
        ValueError: The first message does not have the correct role.
    """
    llm_result = program.metadata['llm_result']
    if llm_result is None:
        logger.warning(
            f"LLM result is None for Gen {program.generation} Branch {program.branch_id}"
            f" - could not extract message history"
        )
        return None

    assert 'new_msg_history' in llm_result

    prompt_msg = llm_result['new_msg_history'][0]
    response_msg = llm_result['new_msg_history'][-1]

    # the following checks are important for debugging -
    # might want to change these checks if the format of message
    # histories changes
    if 'role' not in prompt_msg:
        raise KeyError("expected messages in this program's message history to have a role, found none")
    if 'content' not in prompt_msg:
        raise KeyError("expected messages in this program's message history to have content, found none")
    if prompt_msg['role'] != "user":
        raise ValueError(
            f"expected the first message of this program's message history to have user role, found {prompt_msg['role']}"
        )
    if 'role' not in response_msg:
        raise KeyError("expected messages in this program's message history to have a role, found none")
    if 'content' not in response_msg:
        raise KeyError("expected messages in this program's message history to have content, found none")
    if response_msg['role'] != "assistant":
        logger.warning(
            f"Expected last message of this program's message history to have "
            f"assistant role, found {response_msg['role']} (Gen {program.generation}) Branch {program.branch_id}"
            f" - could not extract response"
        )
        return None

    return Message(**prompt_msg), Message(**response_msg)
