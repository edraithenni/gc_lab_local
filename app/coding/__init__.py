from .base import CodingScheme
from .classic import ClassicGC
from .cce_fr import CCEGCFR
from .cce_cr import CCEGCCR
from .pipelined import PipelinedGC
from .pipelined_fr import PipelinedFR


def make_scheme(config) -> CodingScheme:
    common = dict(
        num_workers=config.num_workers,
        num_partitions=config.num_partitions,
        stragglers=config.stragglers,
    )
    factories = {
        "classic": lambda: ClassicGC(**common, seed=config.gc_seed),
        "cce_gc_fr": lambda: CCEGCFR(
            **common,
            communication_factor=config.communication_factor,
            seed=config.gc_seed,
        ),
        "cce_gc_cr": lambda: CCEGCCR(
            **common, communication_factor=config.communication_factor
        ),
        "pgc_cr": lambda: PipelinedGC(**common),
        "pgc_fr": lambda: PipelinedFR(**common),
    }
    return factories[config.scheme]()
