# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from .metrics import is_sklearn_available
from .processors import (
    DataProcessor,
    InputExample,
    PubmedQAExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    PubmedQAPlusProcessor,
    BioNumFactQAProcessor, 
    BioNumFactQAExample, 
    bionumfactqa_convert_examples_to_features,
    glue_convert_examples_to_features,
    pubmedqa_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
    get_text_num
)


if is_sklearn_available():
    from .metrics import glue_compute_metrics, xnli_compute_metrics
