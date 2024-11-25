from setuptools import find_packages
from setuptools import setup

setup(
    name='turt_q_learn_ros2',
    version='0.0.1',
    packages=find_packages(
        include=('turt_q_learn_ros2', 'turt_q_learn_ros2.*')),
)
