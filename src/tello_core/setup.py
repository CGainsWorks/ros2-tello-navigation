from pathlib import Path

from setuptools import find_packages, setup

package_name = "tello_core"


def data_files():
    """Files installed alongside the package.

    Built rather than hard-coded so that adding a launch file or a config does
    not need this list edited, which is the usual way a launch file ends up
    working from the source tree and missing from the install.
    """
    files = [
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ]
    for folder in ("launch", "config"):
        paths = sorted(str(p) for p in Path(folder).glob("*") if p.is_file())
        if paths:
            files.append((f"share/{package_name}/{folder}", paths))
    return files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CGainsWorks",
    maintainer_email="114424105+CGainsWorks@users.noreply.github.com",
    description="tello_core for the GPS-denied Tello navigation stack",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tello_emulator = tello_core.emulator.__main__:main",
        ],
    },
)
