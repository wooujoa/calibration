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
            'eye_in_hand_calib_any = calibration.eye_in_hand_calib_any:main',
            'eye_in_hand_calib_any_r = calibration.eye_in_hand_calib_any_r:main',
            'eye_in_hand_calib_pub = calibration.eye_in_hand_calib_pub:main',
            'eye_in_hand_calib_pub_point = calibration.eye_in_hand_calib_pub_point:main',
            'eye_in_hand_calib_pub_r = calibration.eye_in_hand_calib_pub_r:main',
            'eye_in_hand_calib_pub_r_point = calibration.eye_in_hand_calib_pub_r_point:main',
            'eye_in_hand_calib_pub_zed_point = calibration.eye_in_hand_calib_pub_zed_point:main',
            'calib_zed = calibration.calib_zed:main',
            'calib_r_any_custom = calibration.calib_r_any_custom:main',
            'CALI_ZED = calibration.CALI_ZED:main',
            'CALI_D405_R = calibration.CALI_D405_R:main',
            'CALI_D405_L = calibration.CALI_D405_L:main',
        ],
    },
)
