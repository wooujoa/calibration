from setuptools import find_packages, setup

package_name = 'calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eye_in_hand_l_calibration = calibration.eye_in_hand_l_calibration:main',
            'eye_in_hand_r_calibration = calibration.eye_in_hand_r_calibration:main',
            'eye_in_hand_calib_any = calibration.eye_in_hand_calib_any:main',
            'eye_in_hand_calib_pub = calibration.eye_in_hand_calib_pub:main',
            'eye_in_hand_calib_pub_point = calibration.eye_in_hand_calib_pub_point:main',
            'eye_in_hand_calib_pub_r = calibration.eye_in_hand_calib_pub_r:main',
            'eye_in_hand_calib_pub_r_point = calibration.eye_in_hand_calib_pub_r_point:main',
            'eye_in_hand_calib_pub_zed_point = calibration.eye_in_hand_calib_pub_zed_point:main',
            'grasp_calib_debug = calibration.grasp_calib_debug:main',
        ],
    },
)
