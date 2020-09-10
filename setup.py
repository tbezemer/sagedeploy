from setuptools import setup, find_packages

requires = [
    "pandas>=1.0.3",
    "scikit-learn>=0.22",
    "scikit-lego>=0.6.0",
    "click>=7.1.2",
    "flask>=1.1.2",
    "gunicorn>=20.0.4"
]

dev_requires = {"dev": ["pytest>=6.0.1"]}

setup(

    name="sagedeploy",
    version="0.0.1",
    author="Tim Bezemer",
    packages=find_packages("src"),
    package_dir={'': 'src'},
    install_requirements=requires,
    extras_require=dev_requires,
    entry_points={
        'console_scripts': [
            'sagedeploy = sagedeploy.cli:main'
        ]
    }

)
