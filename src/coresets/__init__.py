from .lightweight_coreset import LightweightCoreset
from .random_sampling import RandomSampling

ALGORITHMS = {
    "lightweight": LightweightCoreset,
    "random": RandomSampling,
}
