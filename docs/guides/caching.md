# Membership-function cache

highFIS estimators cache the result of membership-function (MF) initialization —
the k-means or grid step run at the start of `fit` — so that repeated `fit` calls
with the same data and hyperparameters skip the recompute.

## Behaviour

- **Process-global and thread-safe.** A single cache is shared by all estimator
  instances in the running Python process.
- **Enabled by default**, with an **LRU** eviction policy and a maximum of **128
  entries**. A cache hit renews the entry, so frequently reused initializations
  are kept.
- **Keyed by** the input data, `mf_init`, `n_mfs`, `sigma_scale`, `random_state`,
  `pfrb_max_rules`, `input_configs` and `rule_base`. Changing any of these
  produces a distinct entry.

The cache only stores the *initialization* of the MFs; it never affects the
trained model, its numerical results, or reproducibility.

## Programmatic control

```python
from highfis import (
    clear_mf_cache,
    mf_cache_info,
    set_mf_cache_enabled,
    set_mf_cache_size,
)

mf_cache_info()              # MFCacheInfo(hits, misses, maxsize, currsize, enabled)
set_mf_cache_size(512)       # change the maximum number of entries (must be >= 1)
set_mf_cache_enabled(False)  # bypass the cache entirely (always rebuild, store nothing)
clear_mf_cache()             # empty the cache and reset the hit/miss counters
```

`mf_cache_info()` returns a named tuple mirroring `functools.lru_cache().cache_info()`
(with an extra `enabled` flag), which is handy for benchmarks and diagnostics.

## Environment variables

The cache can also be configured at import time:

```bash
HIGHFIS_DISABLE_MF_CACHE=1   # disable the cache (truthy: 1/true/yes/on)
HIGHFIS_MF_CACHE_SIZE=512    # maximum number of entries (positive integer)
```

Invalid values are ignored and fall back to the defaults.

## When to disable it

Disabling (or clearing) the cache is useful when sweeping many hyperparameter
combinations in a single process — beyond `maxsize` distinct keys, entries are
evicted continuously and you pay the MF (de)serialization cost without much reuse.
In that scenario, either raise `set_mf_cache_size(...)` or turn the cache off.
