from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wearablehrv",
    version="0.1.13",
    author="Amin Sinichi",
    author_email="aminsinichi@gmail.com",
    description="Wearablehrv: A Python package for the validation of heart rate and heart rate variability in wearables.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AminSinichi/wearable-hrv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "pooch",
        "matplotlib",
        "ipywidgets",
        "IPython",
        "hrv-analysis",
        "avro-python3",
        "pingouin",
        "seaborn",
        "statsmodels",
        "scipy",
        "astropy<6.0",
    ],
)
