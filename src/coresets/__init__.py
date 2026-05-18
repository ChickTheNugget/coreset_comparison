from .lightweight_coreset import LightweightCoreset
from .random_sampling import RandomSampling
from .egb_coreset import EGBCoreset

ALGORITHMS = {
    "lightweight": LightweightCoreset,
    "random": RandomSampling,
    "egb": EGBCoreset,
}
