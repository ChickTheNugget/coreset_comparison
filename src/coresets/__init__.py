from .lightweight_coreset import LightweightCoreset
from .random_sampling import RandomSampling
from .egb_coreset import EGBCoreset
from .egq_coreset import EGQCoreset
from .biased_coreset import BiasedCoreset
from .ke_chen_coreset import KeChenCoreset

ALGORITHMS = {
    "lightweight": LightweightCoreset,
    "random": RandomSampling,
    "egb": EGBCoreset,
    "egq": EGQCoreset,
    "biased": BiasedCoreset,
    "ke_chen": KeChenCoreset
}
