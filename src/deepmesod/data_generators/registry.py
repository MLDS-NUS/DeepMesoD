_GENERATORS = {}


def register_data_generator(name):
    def decorator(value):
        if name in _GENERATORS:
            raise KeyError(f"data generator '{name}' is already registered")
        _GENERATORS[name] = value
        return value

    return decorator


def get_data_generator(name):
    try:
        return _GENERATORS[name]
    except KeyError as error:
        raise KeyError(f"unknown data generator '{name}'") from error


def registered_data_generators():
    return sorted(_GENERATORS)
