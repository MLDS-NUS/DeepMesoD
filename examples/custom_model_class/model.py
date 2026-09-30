from deepmesod.classes import register_dynamical_class
from deepmesod.classes.model_a_scalar import ScalarModelA


@register_dynamical_class("example_scalar_model_a")
class ExampleScalarModelA(ScalarModelA):
    pass
