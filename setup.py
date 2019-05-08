import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setuptools.setup(
        name='ballistico',
        version="0.0.1",
        description='Anharmonic Lattice Dynamics calculator from ASE objects',
        long_description=long_description,
        author='Giuseppe Barbalinardo',
        author_email='giuseppe.barbalinardo@gmail.com',
        url="https://gitlab.com/gbarbalinardo/ballistico",
        license='BSD-3C',
        packages = [
            'ballistico'
            ],
        package_dir = {
            'ballistico': 'ballistico'
            },
        install_requires=[
            'numpy>=1.13',
            'scipy>=1',
            'ase>=3.16.0',
            'sparse>=0.6',
            'spglib>=1.11',
            'seekpath>=1.8',
            'tensorflow>=1.13',
            'opt_einsum>=2.3',
        ],
        zip_safe=True
    )
