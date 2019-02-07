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
        url="https://gitlab.com/gbarbalinardo/finitedifference",
        license='BSD-3C',
        packages = [
            'finitedifference'
            ],
        package_dir = {
            'finitedifference': 'finitedifference'
            },
        install_requires=[
            'numpy>=1.7',
            'scipy>=1',
            'ase>=3.16.0',
            'sparse>=0.6',
            'pandas>=0.23',
            'spglib>=1.11',
            'opt_einsum>=2.3',
            'seaborn>=0.9',
        ],
        zip_safe=True
    )
