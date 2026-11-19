"""LearnByWire shared API; implementations live in separately owned packages."""
import importlib

__version__ = "1.2.2"
_EXPORTS = {
    **{n: "trainerplus" for n in ("TrainerPlus", "TrainingConfig", "DataConfig", "TokenDataset", "TrainResult", "prepare_dataset")},
    **{n: "optimizerplus" for n in ("OptimizerPlus", "TrainingSession", "TokenBatches", "check_environment", "PRODUCT_NAME")},
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    versions = {"__trainer_version__": "trainerplus", "__optimizer_version__": "optimizerplus"}
    module_name = versions.get(name, _EXPORTS.get(name))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        distribution = "learnbywire-" + module_name
        raise ModuleNotFoundError(f"{name} requires {distribution}; install the compatible LearnByWire packages", name=module_name) from error
    return getattr(module, "__version__" if name in versions else name)

def __dir__():
    return sorted(set(globals()) | set(__all__) | {"__trainer_version__", "__optimizer_version__"})
