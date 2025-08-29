import os
import shutil
from setuptools.command.install import install
try:
    from setuptools import setup, find_packages
    use_setuptools = True
    print('setuptools is used')
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print('distutils is used')

class CustomInstallCommand(install):
    def run(self):
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copy2(os.path.join(setup_dir, 'source', 'HTEM.py'), os.path.join(setup_dir, 'HTEM'))
        htem_file = os.path.join(setup_dir, 'HTEM')
        with open(htem_file, 'r', encoding='utf-8') as file:
            content = file.read()
        content = content.replace('config_dir', repr(setup_dir))
        
        with open(htem_file, 'w', encoding='utf-8') as file:
            file.write(content)
        install.run(self)

setup(
    name='HTEM',
    packages=["source"],
    install_requires=['numpy', 'scipy', 'ase', 'spglib', 'matplotlib', 'imageio'],
    author="Zhen Yang",
    author_email="627259879@qq.com",
    scripts=['HTEM','job_sbatch_relax1.sh','job_sbatch_NPT.sh','job_sbatch_strain.sh','HTEM_slurm_sub.sh'],
    zip_safe=False,
    license="LICENSE",
    cmdclass={
        'install': CustomInstallCommand,
    },
)
