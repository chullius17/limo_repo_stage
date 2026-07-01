from setuptools import setup
from glob import glob
import os

package_name = 'traj_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'routes             = traj_package.routes_builder:main',
            'route_combinator   = traj_package.route_combinator:main',
            'astar              = traj_package.astar_server:main',
            'coordinator        = traj_package.coordinator:main'
        ],
    },
)
