import pip

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

links = []
requires = []

try:
    requirements = pip.req.parse_requirements('requirements.txt')
except:
    # new versions of pip requires a session
    requirements = pip.req.parse_requirements(
        'requirements.txt', session=pip.download.PipSession())

for item in requirements:
    if getattr(item, 'url', None):
        links.append(str(item.url))
    if getattr(item, 'link', None):
        links.append(str(item.link))
    if item.req:
        requires.append(str(item.req))

setup(
    name='CM-GP',
    version="0.2.0",
    url='https://github.com/SenneDeproost/CM-GP',
    license='',
    author="Senne Deproost",
    author_email="semme.deproost@vub.be",
    description='Critic-Moderated Genetic Programming',
    long_description="A Genetic Programming library to exploit Deep Reinforcement Learning algorithms.",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=requires,
    dependency_links=links
)