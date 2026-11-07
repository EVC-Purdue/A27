from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

package_name = "autonomous_kart"

NODES = sorted(d.name for d in Path("autonomous_kart/nodes").iterdir() if (d / "__init__.py").exists())

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("autonomous_kart/launch/*.py")),
        ("share/" + package_name + "/params", glob("autonomous_kart/params/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Purdue EVC",
    maintainer_email="shay.manor@gmail.com",
    license="Apache-2.0",
    description="A27 kart nodes, params and launch files.",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"{n}_node = autonomous_kart.nodes.{n}.{n}_node:main" for n in NODES
        ],
    },
)
