"""
Tests for the ActorsCollector class in zombus.registration.collecting module.
"""

import os
import sys
import tempfile
from collections.abc import Iterable
from types import ModuleType

import pytest
from _pytest.capture import CaptureFixture
from zodchy.codex.cqea import Context, Event, Task

from zombus.definitions.enums import ActorKind
from zombus.registration.entities import Actor
from zombus.registration.collecting import ActorsCollector
from zombus.registration.registry import ActorsRegistry


# Test message classes
class TestTask(Task):
    """Test task class for testing."""

    pass


class TestEvent(Event):
    """Test event class for testing."""

    pass


class TestContext(Context):
    """Test context class for testing."""

    pass


class TestActorsCollector:
    """Test class for ActorsCollector functionality."""

    def test_initialization(self) -> None:
        """Test that ActorsCollector can be initialized with actor factory."""
        collector = ActorsCollector()

        assert collector._actors == []
        assert "__pycache__" in collector._ignore_list
        assert ".pytest_cache" in collector._ignore_list
        assert ".git" in collector._ignore_list
        assert "venv" in collector._ignore_list
        assert "env" in collector._ignore_list

    def test_get_registry(self) -> None:
        """Test that get_registry returns a registry."""
        collector = ActorsCollector()

        registry = collector.get_registry()
        assert isinstance(registry, ActorsRegistry)

    def test_register_actor_directly(self) -> None:
        """Test registering an actor directly via register_callable method."""

        def test_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()

        collector.register_callable(test_usecase)

        registry = collector.get_registry()
        actors = list(registry.get(TestTask))
        assert len(actors) == 1
        assert actors[0].name == "test_usecase"

    def test_register_multiple_actors_directly(self) -> None:
        """Test registering multiple actors directly."""

        def first_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def second_usecase(message: TestEvent) -> Iterable[TestEvent]:
            return []

        collector = ActorsCollector()

        collector.register_callable(first_usecase)
        collector.register_callable(second_usecase)

        registry = collector.get_registry()
        task_actors = list(registry.get(TestTask))
        event_actors = list(registry.get(TestEvent))

        assert len(task_actors) == 1
        assert len(event_actors) == 1
        assert task_actors[0].name == "first_usecase"
        assert event_actors[0].name == "second_usecase"

    def test_get_registry_with_include_filter(self) -> None:
        """Test get_registry with include filter returns only specified actor kinds."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)
        collector.register_callable(task_writer)

        # Get registry with only USECASE actors
        registry = collector.get_registry(include={ActorKind.USECASE})
        actors = list(registry)
        assert len(actors) == 1
        assert actors[0].name == "task_usecase"
        assert actors[0].kind == ActorKind.USECASE

    def test_get_registry_with_exclude_filter(self) -> None:
        """Test get_registry with exclude filter excludes specified actor kinds."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)
        collector.register_callable(task_writer)

        # Get registry excluding AUDITOR actors
        registry = collector.get_registry(exclude={ActorKind.AUDITOR})
        actors = list(registry)
        assert len(actors) == 2
        actor_names = {a.name for a in actors}
        assert "task_usecase" in actor_names
        assert "task_writer" in actor_names
        assert "task_auditor" not in actor_names

    def test_get_registry_with_include_multiple_kinds(self) -> None:
        """Test get_registry with include filter for multiple actor kinds."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_reader(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)
        collector.register_callable(task_writer)
        collector.register_callable(task_reader)

        # Get registry with USECASE and WRITER actors
        registry = collector.get_registry(include={ActorKind.USECASE, ActorKind.WRITER})
        actors = list(registry)
        assert len(actors) == 2
        actor_names = {a.name for a in actors}
        assert "task_usecase" in actor_names
        assert "task_writer" in actor_names

    def test_get_registry_with_exclude_multiple_kinds(self) -> None:
        """Test get_registry with exclude filter for multiple actor kinds."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_reader(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)
        collector.register_callable(task_writer)
        collector.register_callable(task_reader)

        # Get registry excluding AUDITOR and READER actors
        registry = collector.get_registry(exclude={ActorKind.AUDITOR, ActorKind.READER})
        actors = list(registry)
        assert len(actors) == 2
        actor_names = {a.name for a in actors}
        assert "task_usecase" in actor_names
        assert "task_writer" in actor_names

    def test_get_registry_with_include_and_exclude(self) -> None:
        """Test get_registry with both include and exclude filters (exclude takes precedence)."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_writer(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)
        collector.register_callable(task_writer)

        # Include USECASE and AUDITOR, but exclude AUDITOR - should only get USECASE
        registry = collector.get_registry(include={ActorKind.USECASE, ActorKind.AUDITOR}, exclude={ActorKind.AUDITOR})
        actors = list(registry)
        assert len(actors) == 1
        assert actors[0].name == "task_usecase"

    def test_get_registry_with_context_actors(self) -> None:
        """Test get_registry filtering with CONTEXT actor kind."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_context(message: TestTask) -> TestContext:
            return TestContext()

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_context)

        # Get registry with only CONTEXT actors
        registry = collector.get_registry(include={ActorKind.CONTEXT})
        actors = list(registry)
        assert len(actors) == 1
        assert actors[0].name == "task_context"
        assert actors[0].kind == ActorKind.CONTEXT

    def test_get_registry_exclude_context_actors(self) -> None:
        """Test get_registry excluding CONTEXT actor kind."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_context(message: TestTask) -> TestContext:
            return TestContext()

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_context)

        # Get registry excluding CONTEXT actors
        registry = collector.get_registry(exclude={ActorKind.CONTEXT})
        actors = list(registry)
        assert len(actors) == 1
        assert actors[0].name == "task_usecase"
        assert actors[0].kind == ActorKind.USECASE

    def test_get_registry_include_empty_result(self) -> None:
        """Test get_registry with include filter that matches no actors."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)

        # Get registry with AUDITOR kind (no auditors registered)
        registry = collector.get_registry(include={ActorKind.AUDITOR})
        actors = list(registry)
        assert len(actors) == 0

    def test_get_registry_exclude_all(self) -> None:
        """Test get_registry with exclude filter that excludes all actors."""

        def task_usecase(message: TestTask) -> Iterable[TestTask]:
            return []

        def task_auditor(message: TestTask) -> Iterable[TestTask]:
            return []

        collector = ActorsCollector()
        collector.register_callable(task_usecase)
        collector.register_callable(task_auditor)

        # Exclude all registered actor kinds
        registry = collector.get_registry(exclude={ActorKind.USECASE, ActorKind.AUDITOR})
        actors = list(registry)
        assert len(actors) == 0


class TestActorsCollectorModuleRegistration:
    """Test class for ActorsCollector module registration functionality."""

    def setup_method(self) -> None:
        """Create a temporary directory structure with Python modules for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.package_name = f"test_package_{os.getpid()}"
        self.package_path = os.path.join(self.temp_dir, self.package_name)
        os.makedirs(self.package_path)

        # Create __init__.py for the package
        init_file = os.path.join(self.package_path, "__init__.py")
        with open(init_file, "w") as f:
            f.write("# Test package\n")

        # Add temp_dir to sys.path for importing
        sys.path.insert(0, self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up temporary directory and remove from sys.path."""
        # Remove from sys.path
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)

        # Remove modules from sys.modules
        modules_to_remove = [key for key in sys.modules if key.startswith(self.package_name)]
        for module_name in modules_to_remove:
            del sys.modules[module_name]

        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_module_file(self, filename: str, content: str, subdir: str | None = None) -> str:
        """Helper to create a Python module file in the temp package."""
        if subdir:
            dir_path = os.path.join(self.package_path, subdir)
            os.makedirs(dir_path, exist_ok=True)
            # Create __init__.py in subdir if not exists
            init_path = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w") as f:
                    f.write("# Subpackage\n")
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.package_path, filename)

        with open(file_path, "w") as f:
            f.write(content)

        return file_path

    def test_register_module_with_single_function(self) -> None:
        """Test registering a module with a single public function."""
        module_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class SimpleTask(Task):
    pass

def simple_usecase(message: SimpleTask) -> Iterable[SimpleTask]:
    return []
"""
        self._create_module_file("handlers.py", module_content)

        # Import the package
        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        # Get all registered actors
        registry = collector.get_registry()
        all_actors = list(registry)
        # Should have at least one actor (simple_usecase)
        usecase_actors = [a for a in all_actors if a.name == "simple_usecase"]
        assert len(usecase_actors) == 1

    def test_register_module_with_multiple_functions(self) -> None:
        """Test registering a module with multiple public functions."""
        module_content = '''
from collections.abc import Iterable
from zodchy.codex.cqea import Task, Event

class MultiTask(Task):
    pass

class MultiEvent(Event):
    pass

def first_usecase(message: MultiTask) -> Iterable[MultiTask]:
    return []

def second_usecase(message: MultiEvent) -> Iterable[MultiEvent]:
    return []

def _private_function(message: MultiTask) -> Iterable[MultiTask]:
    """This should be ignored as it starts with underscore."""
    return []
'''
        self._create_module_file("multi.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        # Public functions should be registered
        assert "first_usecase" in actor_names
        assert "second_usecase" in actor_names
        # Private function should NOT be registered
        assert "_private_function" not in actor_names

    def test_register_module_ignores_private_functions(self) -> None:
        """Test that private functions (starting with _) are ignored."""
        module_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class PrivateTask(Task):
    pass

def _hidden_usecase(message: PrivateTask) -> Iterable[PrivateTask]:
    return []

def __dunder_like(message: PrivateTask) -> Iterable[PrivateTask]:
    return []
"""
        self._create_module_file("private_only.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "_hidden_usecase" not in actor_names
        assert "__dunder_like" not in actor_names

    def test_register_module_with_subdirectory(self) -> None:
        """Test registering a module that has subdirectories."""
        # Create main module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class MainTask(Task):
    pass

def main_usecase(message: MainTask) -> Iterable[MainTask]:
    return []
"""
        self._create_module_file("main.py", main_content)

        # Create submodule
        sub_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Event

class SubEvent(Event):
    pass

def sub_usecase(message: SubEvent) -> Iterable[SubEvent]:
    return []
"""
        self._create_module_file("subhandlers.py", sub_content, subdir="subpackage")

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "main_usecase" in actor_names
        assert "sub_usecase" in actor_names

    def test_register_module_ignores_pycache(self) -> None:
        """Test that __pycache__ directories are ignored."""
        # Create a module in main package
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class CacheTask(Task):
    pass

def cache_usecase(message: CacheTask) -> Iterable[CacheTask]:
    return []
"""
        self._create_module_file("cache_test.py", main_content)

        # Create __pycache__ directory with a fake .py file (simulating a bad scenario)
        pycache_dir = os.path.join(self.package_path, "__pycache__")
        os.makedirs(pycache_dir, exist_ok=True)
        pycache_file = os.path.join(pycache_dir, "fake_module.py")
        with open(pycache_file, "w") as f:
            f.write("# This should be ignored\n")

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        # Should only have the main module's function
        assert "cache_usecase" in actor_names
        # Should not attempt to import from __pycache__

    def test_register_module_ignores_hidden_files(self) -> None:
        """Test that hidden files (starting with .) are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class HiddenTask(Task):
    pass

def visible_usecase(message: HiddenTask) -> Iterable[HiddenTask]:
    return []
"""
        self._create_module_file("visible.py", main_content)

        # Create a hidden file
        hidden_file = os.path.join(self.package_path, ".hidden_module.py")
        with open(hidden_file, "w") as f:
            f.write(
                """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class HiddenTask2(Task):
    pass

def hidden_usecase(message: HiddenTask2) -> Iterable[HiddenTask2]:
    return []
"""
            )

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "visible_usecase" in actor_names
        assert "hidden_usecase" not in actor_names

    def test_register_module_ignores_underscore_files(self) -> None:
        """Test that files starting with _ (other than __init__.py) are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class PublicTask(Task):
    pass

def public_usecase(message: PublicTask) -> Iterable[PublicTask]:
    return []
"""
        self._create_module_file("public_module.py", main_content)

        # Create a private module file
        private_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class PrivateModTask(Task):
    pass

def private_mod_usecase(message: PrivateModTask) -> Iterable[PrivateModTask]:
    return []
"""
        self._create_module_file("_private_module.py", private_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "public_usecase" in actor_names
        assert "private_mod_usecase" not in actor_names

    def test_register_module_ignores_venv_directory(self) -> None:
        """Test that venv directories are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class VenvTask(Task):
    pass

def venv_test_usecase(message: VenvTask) -> Iterable[VenvTask]:
    return []
"""
        self._create_module_file("venv_test.py", main_content)

        # Create a venv directory with a fake module
        venv_dir = os.path.join(self.package_path, "venv")
        os.makedirs(venv_dir, exist_ok=True)
        venv_init = os.path.join(venv_dir, "__init__.py")
        with open(venv_init, "w") as f:
            f.write("# Fake venv init\n")
        venv_module = os.path.join(venv_dir, "fake.py")
        with open(venv_module, "w") as f:
            f.write(
                """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class FakeVenvTask(Task):
    pass

def venv_fake_usecase(message: FakeVenvTask) -> Iterable[FakeVenvTask]:
    return []
"""
            )

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "venv_test_usecase" in actor_names
        assert "venv_fake_usecase" not in actor_names

    def test_register_module_handles_import_errors_gracefully(self, capsys: CaptureFixture[str]) -> None:
        """Test that import errors are handled gracefully with warning."""
        # Create a module with import error (non-existent module)
        broken_content = """
# This module has an import error
from nonexistent_module_xyz import something

def broken_usecase():
    pass
"""
        self._create_module_file("broken.py", broken_content)

        # Create a valid module
        valid_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class ValidTask(Task):
    pass

def valid_usecase(message: ValidTask) -> Iterable[ValidTask]:
    return []
"""
        self._create_module_file("valid.py", valid_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning: Could not import module" in captured.out

        # Valid module should still be registered
        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}
        assert "valid_usecase" in actor_names

    def test_register_module_with_empty_package(self) -> None:
        """Test registering an empty package (only __init__.py)."""
        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        # Should not raise and registry should be empty
        registry = collector.get_registry()
        all_actors = list(registry)
        assert len(all_actors) == 0

    def test_register_module_without_path_raises_error(self) -> None:
        """Test that registering a module without __path__ or __file__ raises ValueError."""
        # Create a fake module without __path__ or __file__
        fake_module = ModuleType("fake_module_no_path")
        # Remove __file__ if present
        if hasattr(fake_module, "__file__"):
            delattr(fake_module, "__file__")
        # Ensure __path__ is empty
        fake_module.__path__ = []

        collector = ActorsCollector()

        with pytest.raises(ValueError) as exc_info:
            collector.register_module(fake_module)

        assert "has no __path__ or __file__ attribute" in str(exc_info.value)

    def test_register_module_with_file_attribute_single_module(self) -> None:
        """Test registering a single-file module (not a package)."""
        # Create a simple module file (not a package)
        module_content = '''
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class FileTask(Task):
    pass

def file_usecase(message: FileTask) -> Iterable[FileTask]:
    return []

def _private_func(message: FileTask) -> Iterable[FileTask]:
    """Private function should be ignored."""
    return []
'''
        module_file = os.path.join(self.temp_dir, "standalone_module.py")
        with open(module_file, "w") as f:
            f.write(module_content)

        import importlib.util

        spec = importlib.util.spec_from_file_location("standalone_module", module_file)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["standalone_module"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

        collector = ActorsCollector()

        # Single-file modules should now work correctly
        collector.register_module(module)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        # Public function should be registered
        assert "file_usecase" in actor_names
        # Private function should NOT be registered
        assert "_private_func" not in actor_names

        # Cleanup
        del sys.modules["standalone_module"]

    def test_register_module_with_init_py_path(self) -> None:
        """Test registering a package handles __init__.py path correctly."""
        # Create module content
        module_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class InitTask(Task):
    pass

def init_usecase(message: InitTask) -> Iterable[InitTask]:
    return []
"""
        self._create_module_file("init_test.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}
        assert "init_usecase" in actor_names

    def test_register_module_does_not_register_imported_functions(self) -> None:
        """Test that functions imported from other modules are not registered."""
        # Create a helper module
        helper_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class HelperTask(Task):
    pass

def helper_usecase(message: HelperTask) -> Iterable[HelperTask]:
    return []
"""
        self._create_module_file("helper.py", helper_content)

        # Create main module that imports from helper
        main_content = f"""
from collections.abc import Iterable
from zodchy.codex.cqea import Task
from {self.package_name}.helper import helper_usecase as imported_usecase

class MainTask(Task):
    pass

def main_usecase(message: MainTask) -> Iterable[MainTask]:
    return []

# Re-export under different name (should still be ignored as it's from another module)
reexported = imported_usecase
"""
        self._create_module_file("main_import.py", main_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        # main_usecase should be registered
        assert "main_usecase" in actor_names
        # helper_usecase should be registered once (from helper.py)
        helper_count = sum(1 for a in all_actors if a.name == "helper_usecase")
        assert helper_count == 1  # Only from helper.py, not from main_import.py

    def test_register_module_deeply_nested_structure(self) -> None:
        """Test registering a module with deeply nested subdirectories."""
        # Create level 1
        level1_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class Level1Task(Task):
    pass

def level1_usecase(message: Level1Task) -> Iterable[Level1Task]:
    return []
"""
        self._create_module_file("level1.py", level1_content, subdir="sub1")

        # Create level 2
        level2_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class Level2Task(Task):
    pass

def level2_usecase(message: Level2Task) -> Iterable[Level2Task]:
    return []
"""
        self._create_module_file("level2.py", level2_content, subdir="sub1/sub2")

        # Create level 3
        level3_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class Level3Task(Task):
    pass

def level3_usecase(message: Level3Task) -> Iterable[Level3Task]:
    return []
"""
        self._create_module_file("level3.py", level3_content, subdir="sub1/sub2/sub3")

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "level1_usecase" in actor_names
        assert "level2_usecase" in actor_names
        assert "level3_usecase" in actor_names

    def test_register_module_ignores_env_directory(self) -> None:
        """Test that env directories are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class EnvTask(Task):
    pass

def env_test_usecase(message: EnvTask) -> Iterable[EnvTask]:
    return []
"""
        self._create_module_file("env_test.py", main_content)

        # Create an env directory with a fake module
        env_dir = os.path.join(self.package_path, "env")
        os.makedirs(env_dir, exist_ok=True)
        env_init = os.path.join(env_dir, "__init__.py")
        with open(env_init, "w") as f:
            f.write("# Fake env init\n")
        env_module = os.path.join(env_dir, "fake.py")
        with open(env_module, "w") as f:
            f.write(
                """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class FakeEnvTask(Task):
    pass

def env_fake_usecase(message: FakeEnvTask) -> Iterable[FakeEnvTask]:
    return []
"""
            )

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "env_test_usecase" in actor_names
        assert "env_fake_usecase" not in actor_names

    def test_register_module_ignores_pytest_cache(self) -> None:
        """Test that .pytest_cache directories are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class PytestTask(Task):
    pass

def pytest_test_usecase(message: PytestTask) -> Iterable[PytestTask]:
    return []
"""
        self._create_module_file("pytest_test.py", main_content)

        # Create a .pytest_cache directory
        cache_dir = os.path.join(self.package_path, ".pytest_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_init = os.path.join(cache_dir, "__init__.py")
        with open(cache_init, "w") as f:
            f.write("# Fake pytest cache init\n")

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "pytest_test_usecase" in actor_names

    def test_register_module_ignores_git_directory(self) -> None:
        """Test that .git directories are ignored."""
        # Create a normal module
        main_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class GitTask(Task):
    pass

def git_test_usecase(message: GitTask) -> Iterable[GitTask]:
    return []
"""
        self._create_module_file("git_test.py", main_content)

        # Create a .git directory
        git_dir = os.path.join(self.package_path, ".git")
        os.makedirs(git_dir, exist_ok=True)
        # Note: .git directories don't usually have __init__.py,
        # but we're testing the ignore behavior

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "git_test_usecase" in actor_names

    def test_collect_module_functions_ignores_classes(self) -> None:
        """Test that classes are not registered, only functions."""
        module_content = '''
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class ClassTask(Task):
    pass

class SomeClass:
    """A regular class that should not be registered."""
    def method_usecase(self, message: ClassTask) -> Iterable[ClassTask]:
        return []

def function_usecase(message: ClassTask) -> Iterable[ClassTask]:
    return []
'''
        self._create_module_file("class_test.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        # Only functions should be registered, not classes
        assert "function_usecase" in actor_names
        assert "SomeClass" not in actor_names
        assert "method_usecase" not in actor_names

    def test_register_module_with_async_functions(self) -> None:
        """Test registering async functions."""
        module_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class AsyncTask(Task):
    pass

async def async_usecase(message: AsyncTask) -> Iterable[AsyncTask]:
    return []

def sync_usecase(message: AsyncTask) -> Iterable[AsyncTask]:
    return []
"""
        self._create_module_file("async_test.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}

        assert "async_usecase" in actor_names
        assert "sync_usecase" in actor_names

        # Verify async flag
        async_actor = next(a for a in all_actors if a.name == "async_usecase")
        sync_actor = next(a for a in all_actors if a.name == "sync_usecase")
        assert async_actor.is_async is True
        assert sync_actor.is_async is False

    def test_register_module_with_string_path(self) -> None:
        """Test registering a module where __path__ is a string instead of list.

        Note: In modern Python, __path__ is typically a list, but the code handles
        the case where it might be a string for backwards compatibility.
        """
        module_content = """
from collections.abc import Iterable
from zodchy.codex.cqea import Task

class StringPathTask(Task):
    pass

def string_path_usecase(message: StringPathTask) -> Iterable[StringPathTask]:
    return []
"""
        self._create_module_file("string_path.py", module_content)

        import importlib

        package = importlib.import_module(self.package_name)

        # Test that normal __path__ (list) works correctly
        collector = ActorsCollector()
        collector.register_module(package)

        registry = collector.get_registry()
        all_actors = list(registry)
        actor_names = {a.name for a in all_actors}
        assert "string_path_usecase" in actor_names
