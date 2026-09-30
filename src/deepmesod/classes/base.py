from abc import ABC, abstractmethod

import torch


class DynamicalClass(ABC):
    name = ""

    @abstractmethod
    def build_drift_model(self, config):
        raise NotImplementedError

    @abstractmethod
    def build_factor(self, config):
        raise NotImplementedError

    @abstractmethod
    def build_thermodynamic_model(self, config):
        raise NotImplementedError

    @abstractmethod
    def force(self, thermodynamics, field, *, create_graph):
        raise NotImplementedError

    @abstractmethod
    def drift(self, factor, thermodynamics, field, beta, coarse_volume):
        raise NotImplementedError

    def validate_field(self, field):
        if not torch.is_floating_point(field):
            raise TypeError("field must have a floating dtype")
