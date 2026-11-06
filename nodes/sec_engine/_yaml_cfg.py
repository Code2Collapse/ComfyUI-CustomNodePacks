"""C2C (L7.90): YAML loading for the SAM 2 configs without omegaconf.

PyYAML's default resolver reads "1e-6" (no dot) as a STRING; OmegaConf reads it as a float. This loader adds the
YAML 1.2 float pattern so the configs load exactly as OmegaConf loaded them."""
import re

import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)  # noqa: S506 - _Loader is a SafeLoader subclass
