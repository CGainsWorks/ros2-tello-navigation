"""The ROS 2 packages, checked without a ROS 2 installation.

The nodes cannot be imported here: they import ``rclpy`` at module scope, as ROS
nodes do and this repository's whole point is that everything of substance runs
without it. What *can* be checked and is worth checking on every push, is that
the packaging is coherent. Nearly every "it works on my machine" failure in a ROS
workspace is one of these:

* a ``package.xml`` that does not name the package its directory does;
* a console script pointing at a module or a function that does not exist;
* a node that stopped parsing;
* a launch file referring to a package that is not a dependency.

All four are static properties of the tree and all four are found here in
milliseconds instead of after a two-minute colcon build on someone else's laptop.
"""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

PYTHON_PACKAGES = [
    "tello_core",
    "tello_driver",
    "tello_localisation",
    "tello_control",
    "tello_bringup",
]
ALL_PACKAGES = PYTHON_PACKAGES + ["tello_msgs"]


def _src(repo_root: Path) -> Path:
    return repo_root / "src"


@pytest.mark.parametrize("name", ALL_PACKAGES)
def test_package_xml_is_wellformed_and_names_itself(repo_root, name):
    path = _src(repo_root) / name / "package.xml"
    assert path.is_file(), f"{name} has no package.xml"
    root = ElementTree.parse(path).getroot()
    assert root.findtext("name") == name
    assert root.findtext("license") == "MIT"
    assert root.findtext("version")
    assert (root.findtext("description") or "").strip()


@pytest.mark.parametrize("name", PYTHON_PACKAGES)
def test_python_packages_declare_ament_python(repo_root, name):
    root = ElementTree.parse(_src(repo_root) / name / "package.xml").getroot()
    export = root.find("export")
    assert export is not None
    assert export.findtext("build_type") == "ament_python"


def test_message_package_declares_ament_cmake(repo_root):
    root = ElementTree.parse(_src(repo_root) / "tello_msgs" / "package.xml").getroot()
    assert root.find("export").findtext("build_type") == "ament_cmake"


@pytest.mark.parametrize("name", PYTHON_PACKAGES)
def test_ament_resource_marker_exists(repo_root, name):
    """Without this file the package builds and is then invisible to ``ros2 run``."""
    assert (_src(repo_root) / name / "resource" / name).is_file()


@pytest.mark.parametrize("name", [p for p in PYTHON_PACKAGES if p != "tello_bringup"])
def test_python_module_is_a_package_setuptools_can_find(repo_root, name):
    """``find_packages()`` skips a directory without ``__init__.py``.

    Three of the node packages shipped without one. Every check in this file
    passed, colcon built them, installed their console scripts and installed no
    code, so every node died on import with ``ModuleNotFoundError``. It was found
    the first time the graph was launched against the emulator.
    """
    assert (_src(repo_root) / name / name / "__init__.py").is_file()


@pytest.mark.parametrize("name", PYTHON_PACKAGES)
def test_setup_py_parses_and_names_the_package(repo_root, name):
    path = _src(repo_root) / name / "setup.py"
    assert path.is_file()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "package_name" for t in node.targets)
    ]
    assert assignments, f"{name}/setup.py does not define package_name"
    assert ast.literal_eval(assignments[0].value) == name


def _console_scripts(setup_py: Path):
    """Pull the console_scripts list out of setup.py without executing it."""
    tree = ast.parse(setup_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "console_scripts":
                return [ast.literal_eval(element) for element in value.elts]
    return []


@pytest.mark.parametrize("name", PYTHON_PACKAGES)
def test_every_console_script_points_at_something_real(repo_root, name):
    package = _src(repo_root) / name
    for entry in _console_scripts(package / "setup.py"):
        _, _, target = entry.partition("=")
        module_path, _, function = target.strip().partition(":")
        relative = Path(*module_path.split(".")).with_suffix(".py")
        module_file = package / relative
        assert module_file.is_file(), f"{entry}: {module_file} does not exist"

        tree = ast.parse(module_file.read_text(encoding="utf-8"))
        names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert function in names, f"{entry}: {module_file} defines no {function}()"


def test_the_expected_nodes_are_all_registered(repo_root):
    """A node nobody can launch is a node that does not exist."""
    registered = set()
    for name in PYTHON_PACKAGES:
        for entry in _console_scripts(_src(repo_root) / name / "setup.py"):
            registered.add(entry.split("=")[0].strip())
    assert {"driver_node", "marker_node", "fusion_node", "mission_node"} <= registered


def _node_files(repo_root: Path):
    for name in PYTHON_PACKAGES:
        for path in (_src(repo_root) / name).rglob("*_node.py"):
            yield path


def test_every_node_parses_and_has_a_main(repo_root):
    files = list(_node_files(repo_root))
    assert files, "no ROS 2 nodes found"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert "main" in functions, f"{path} has no main()"


def test_nodes_put_the_core_package_on_the_path_before_importing_it(repo_root):
    """The nodes import ``tello_core`` from the sibling source directory.

    That is what lets a colcon build and a bare pytest run see the same source
    with no copy step and it only works if the path insertion happens before the
    import. Checked structurally rather than by importing, since importing needs
    rclpy.
    """
    for path in _node_files(repo_root):
        source = path.read_text(encoding="utf-8")
        if "tello_core" not in source:
            continue
        insert_at = source.index("sys.path.insert")
        import_at = source.index("from tello_core")
        assert insert_at < import_at, f"{path} imports tello_core before extending sys.path"


def test_message_definitions_exist_and_are_listed_in_cmake(repo_root):
    package = _src(repo_root) / "tello_msgs"
    definitions = sorted(p.name for p in (package / "msg").glob("*.msg"))
    assert definitions
    cmake = (package / "CMakeLists.txt").read_text(encoding="utf-8")
    for definition in definitions:
        assert f"msg/{definition}" in cmake, f"{definition} is not in CMakeLists.txt"


def test_message_fields_have_types_and_names(repo_root):
    for path in (_src(repo_root) / "tello_msgs" / "msg").glob("*.msg"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) >= 2, f"{path.name}: cannot parse {line!r}"


def test_launch_files_only_use_declared_dependencies(repo_root):
    bringup = _src(repo_root) / "tello_bringup"
    declared = {
        element.text
        for element in ElementTree.parse(bringup / "package.xml").getroot()
        if element.text
    }
    for path in (bringup / "launch").glob("*.launch.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "package":
                continue
            if isinstance(node.value, ast.Constant):
                assert node.value.value in declared, (
                    f"{path.name} launches {node.value.value}, which "
                    "tello_bringup/package.xml does not depend on"
                )
