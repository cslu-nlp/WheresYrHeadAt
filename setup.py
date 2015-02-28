from setuptools import setup


setup(name="WheresYrHeadAt",
      version="0.2",
      description="'Where's Yr Head At', a greedy dependency parser",
      author="Kyle Gorman",
      author_email="gormanky@ohsu.edu",
      url="http://github.com/cslu-nlp/WheresYrHeadAt/",
      install_requires=["nlup >= 0.4.0", "nltk >= 3.0.0"],
      dependency_links=["http://github.com/cslu-nlp/nlup/archive/master.zip#egg=nlup-0.4.0"],
      packages=["wheresyrheadat"])
