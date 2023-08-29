from setuptools import setup, find_packages

setup(
    name="wolfpack",
    version="1.0.0",
    description="Wolfpack environment",
    author="David Rother",
    url="https://github.com/semitable/lb-foraging",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10"
    ],
    install_requires=["gym", "gymnasium", "pettingzoo", "matplotlib"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
