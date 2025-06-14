from setuptools import setup, find_packages

setup(
    name="medgemma-brain",
    version="0.1",
    package_dir={"": "src"},  # Tell setuptools to look in src/
    packages=find_packages(where="src"),  # Find packages in src/
    install_requires=[
        "kagglehub",
        "torch",
        "transformers",
        "pillow",
        "scikit-learn",
        "python-dotenv"
    ],
)