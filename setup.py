from setuptools import find_packages, setup

setup(
    name="bandits",
    description="Multi-Armed Bandits Framework",
    packages=find_packages(),
    install_requires=["numpy==1.20.3", "seaborn==0.11.0", "matplotlib==3.4.2"],
    tests_require=["pytest"],
)
