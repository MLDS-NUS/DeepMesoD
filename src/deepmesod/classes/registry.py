import importlib
import importlib.util
from pathlib import Path

_DYNAMICAL_CLASSES = {}
_TRAINERS = {}
_LOSSES = {}


def _register(registry, name, value):
    if not name or not isinstance(name, str):
        raise ValueError("registry name must be a nonempty string")
    if name in registry:
        raise KeyError(f"'{name}' is already registered")
    registry[name] = value
    return value


def register_dynamical_class(name):
    def decorator(value):
        return _register(_DYNAMICAL_CLASSES, name, value)

    return decorator


def register_trainer(name):
    def decorator(value):
        return _register(_TRAINERS, name, value)

    return decorator


def register_loss(name):
    def decorator(value):
        return _register(_LOSSES, name, value)

    return decorator


def get_dynamical_class(name):
    try:
        return _DYNAMICAL_CLASSES[name]
    except KeyError as error:
        raise KeyError(f"unknown dynamical class '{name}'") from error


def get_trainer(name):
    try:
        return _TRAINERS[name]
    except KeyError as error:
        raise KeyError(f"unknown trainer '{name}'") from error


def get_loss(name):
    try:
        return _LOSSES[name]
    except KeyError as error:
        raise KeyError(f"unknown loss '{name}'") from error


def registered_components():
    return {
        "dynamical_classes": sorted(_DYNAMICAL_CLASSES),
        "trainers": sorted(_TRAINERS),
        "losses": sorted(_LOSSES),
    }


def load_extensions(entries, base_dir=None):
    base_dir = Path(base_dir or ".")
    for entry in entries or ():
        candidate = Path(entry)
        if candidate.suffix == ".py" or candidate.parts[:1] in ((".",), ("..",)):
            path = candidate if candidate.is_absolute() else base_dir / candidate
            path = path.resolve()
            name = f"deepmesod_extension_{path.stem}_{abs(hash(path))}"
            specification = importlib.util.spec_from_file_location(name, path)
            if specification is None or specification.loader is None:
                raise ImportError(f"cannot load extension from '{path}'")
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
        else:
            importlib.import_module(entry)
