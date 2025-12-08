import importlib
import inspect
import os
from types import ModuleType

from zombus.definitions.contracts import ActorCallableContract
from zombus.definitions.enums import ActorKind
from zombus.registration.entities import Actor

from .registry import ActorsRegistry


class ActorsCollector:
    def __init__(self) -> None:
        self._actors: list[Actor] = []
        self._ignore_list = ["__pycache__", ".pytest_cache", ".git", "venv", "env"]

    def register_module(self, package: ModuleType) -> None:
        """
        registers all callable objects from the given package and its submodules in the registry.
        Supports both packages (directories with __init__.py) and single-file modules.
        @param package: The package or module to register callables from
        """
        package_path = getattr(package, "__path__", None)
        file_path = getattr(package, "__file__", None)

        # Handle packages (have __path__)
        if package_path:
            base_path = package_path[0] if isinstance(package_path, list) else package_path
            base_path = os.path.dirname(base_path) if base_path.endswith("__init__.py") else base_path
            self._walk_filesystem(base_path, package.__name__)
        # Handle single-file modules (have __file__ but no __path__)
        elif file_path:
            self._collect_module_functions(package)
        else:
            raise ValueError(f"Module {package.__name__} has no __path__ or __file__ attribute")

    def register_callable(self, actor_callable: ActorCallableContract) -> None:
        """
        registers a callable object in the registry
        @param callable_obj: The callable object to register
        """
        self._actors.append(Actor(actor_callable))

    def get_registry(
        self, include: set[ActorKind] | None = None, exclude: set[ActorKind] | None = None
    ) -> ActorsRegistry:
        """
        returns the underlying actors registry
        @return: The actors registry
        """
        registry = ActorsRegistry()
        for actor in self._actors:
            if exclude and actor.kind in exclude:
                continue
            if include and actor.kind not in include:
                continue
            registry.register(actor)
        return registry

    def _walk_filesystem(self, dir_path: str, package_name: str) -> None:
        """
        Recursively walks through the filesystem starting from dir_path and collects functions from Python modules.
        @param dir_path: The directory path to start the walk.
        @param package_name: The base package name corresponding to dir_path.
        """
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)

            # bypass hidden files and directories
            if item.startswith("."):
                continue

            # if it's a directory
            if os.path.isdir(item_path):
                # bypass ignored directories
                if item in self._ignore_list:
                    continue

                # recurse into the directory
                new_package_name = f"{package_name}.{item}"
                self._walk_filesystem(item_path, new_package_name)

            # if its a python file (excluding private/special files)
            elif item.endswith(".py") and not item.startswith("_"):
                module_name = item[:-3]  # Remove .py extension
                full_module_name = f"{package_name}.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)
                    self._collect_module_functions(module)
                except (ImportError, AttributeError, TypeError) as e:
                    print(f"Warning: Could not import module {full_module_name}: {e}")
                    continue

    def _collect_module_functions(self, module: ModuleType) -> None:
        """
        Collects all functions defined in the given module and registers them.
        @param module: The module to inspect.
        """
        module_name = module.__name__

        for name, obj in inspect.getmembers(module):
            # bypass private functions
            if name.startswith("_"):
                continue

            # check if the object is a function defined in this module
            if (
                inspect.isfunction(obj)
                and not inspect.isclass(obj)
                and hasattr(obj, "__module__")
                and obj.__module__ == module_name
            ):
                self._actors.append(Actor(obj))
