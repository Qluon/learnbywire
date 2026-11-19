"""Optional public OptimizerPlus 1.2.2 contract; no proprietary implementation."""
import importlib


def load_optimizerplus():
    try:
        module = importlib.import_module("optimizerplus")
    except ModuleNotFoundError as error:
        if error.name != "optimizerplus":
            raise
        raise ModuleNotFoundError("OptimizerPlus control requires an authorized OptimizerPlus 1.2.2 installation",
                                  name="optimizerplus") from error
    if getattr(module, "__version__", None) != "1.2.2":
        raise RuntimeError("This integration requires OptimizerPlus 1.2.2")
    return module
