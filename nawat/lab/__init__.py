"""The lab — experiments assembled from a fixed backbone and swappable blocks.

    nawat preset ls                         what can be trained
    nawat exp plan llm-grpo --set train.learning_rate=5e-6
    nawat exp approve llm-grpo/20261102-ab12      at a terminal, after reading the table
    nawat exp run llm-grpo/20261102-ab12
    nawat exp report llm-grpo/20261102-ab12       text for a person or an agent

See :mod:`nawat.lab.blocks` for what a block is and
:mod:`nawat.lab.experiment` for the folder an experiment lives in.
"""

from .params import P

__all__ = ["P"]
