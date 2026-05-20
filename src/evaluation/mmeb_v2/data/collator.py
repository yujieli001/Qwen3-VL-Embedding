import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEvalDataCollator:
    """Collator for MMEB evaluation datasets.

    Prepares batches for query or candidate encoding during evaluation.

    Args:
        encode_side: 'qry' for query encoding, 'cand' for candidate encoding.

    Returns:
        Tuple of (batch_inputs, dataset_infos):
        - batch_inputs: List of multimodal input dicts
        - dataset_infos: List of metadata dicts
    """
    encode_side: str

    def __call__(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """Collate a batch of examples.

        Args:
            examples: List of dataset samples, each containing 'query_input' or
                     'cand_input' and 'dataset_infos'.

        Returns:
            Tuple containing:
            - batch_inputs: List of multimodal input dicts for the model
            - dataset_infos: List of metadata (labels, candidate names, etc.)
        """
        # Select input key based on encoding side
        input_key = "query_input" if self.encode_side == 'qry' else "cand_input"

        # Extract batch inputs: List[List[Dict]]
        batch_inputs = [ex[input_key] for ex in examples]

        # Extract metadata
        dataset_infos = [ex["dataset_infos"] for ex in examples]

        return batch_inputs, dataset_infos