from setuptools import setup, find_packages


setup(
    name="DataGossip",
    version="0.4.2",
    description="Distributed Machine Learning Training through Reactive Data Exchange",
    author="Phillip Wenig",
    author_email="phillip.wenig@hpi.de",
    url="https://hpi.de/naumann/people/phillip-wenig.html",
    packages=find_packages(),
    test_suite="tests"
)