

# Parking forecasting

> Parking forecasting pipeline for prediction of available parking spaces

## To use with docker

docker build -t forecast .  
docker run --name parking -d forecast  
docker logs -f [Container ID]

<!-- ## Setup
1. Install git and checkout the [git code repository]
2. Install [anaconda] python version 3.6+
3. Change working directory into the git code repository root
4. Create the self contained conda environment. In a terminal go to the git code repository root and enter the command:

   `conda env create --file conda_env.yml`

5. Any python modules under src need to be available to other scripts. This can be done in a couple of ways. You can
setup and install the python modules by executing the setup.py command below which will install the packages to the
conda environments site-packages folder but with a symlink to the src folder so modifications are reflected immediately.

   `python setup.py develop`

    As an alternative you may prefer to set the python path directly from the console, within notebooks, test scripts
    etc. From Pycharm you can also right click the src folder and select the _Mark Directory As | Source Root_ option.

6. .. Place your own project specific setup steps here e.g. copying data files ...

When distributing your module, you can create a Python egg with the command `python setup.py bdist_egg` and upload the egg.

NOTE: When working in the project notebooks from within the Equinor network, you may need to include the lines below if your proxy is not otherwise setup.

`os.environ['HTTP_PROXY']="http://www-proxy.statoil.no:80"`<br />
`os.environ['HTTPS_PROXY']="http://www-proxy.statoil.no:80"`

## Using the Python Conda environment

Once the Python Conda environment has been set up, you can

* Activate the environment using the following command in a terminal window:

  * Windows: `activate conda_env`
  * Linux, OS X: `source activate conda_env`
  * The __environment is activated per terminal session__, so you must activate it every time you open terminal.

* Deactivate the environment using the following command in a terminal window:

  * Windows: `deactivate conda_env`
  * Linux, OS X: `source deactivate conda_env`

* Delete the environment using the command (can't be undone):

  * `conda remove --name conda_env --all` -->

## File Structure

```
├── .gitignore                <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if
│                                needed
|
├── LICENSE
|
├── README.md                 <- The top-level README for developers using this project.
|
├── setup.py                  <- Metadata about your project for easy distribution.
│
├── requirements.txt          <- The requirements file for reproducing the analysis environments
│                              
├── conda_env.yml             <- Conda environment definition for ensuring consistent setup across environments
|
├── logs                      <- ML model logs
|
├── data
│   ├── interim_[desc]        <- Interim files - give these folders whatever name makes sense.
│   ├── processed             <- The final, canonical data sets for modeling.
│   ├── raw                   <- The original, immutable data dump.
│   ├── temp                  <- Temporary files.
│   └── training              <- Files relating to the training process
│
│
├── src                       <- Code for use in this project.
│   └── PMV4Cast              <- Example python package - place shared code in such a package
│       ├── __init__.py       <- Python package initialisation
│       ├── examplemodule.py  <- Example module with functions and naming / commenting best practices
│       ├── features.py       <- Feature engineering functionality
│       ├── io_data.py        <- IO functionality
│       ├── ml_model.py       <- Machine learning model
│       ├── simple_average.py <- Baseline average forecasting model (next day with 10 min resolution)
│       └── pipeline.py       <- Pipeline functionality
│
└── tests                     <- Test cases (named after module)
    └── PMV4Cast              <- PMV4Cast tests
        ├── examplemodule     <- examplemodule tests (1 file per method tested)
        ├── features          <- features tests
        ├── io                <- io tests
        └── pipeline          <- pipeline tests
```
<!--
## MLOps
Starter scripts for MLOps with Azure ML Service are included as a part of this template in the scripts folder and may be
customised for your own purposes. Please browse the contents of the scripts folder for more details.

For model training, the provided setup allows for running locally without any dependency on Azure ML by running train.py
in the scripts/train folder directly. Alternatively you can submit local or remote runs using the submit scripts in the
same folder.

## Testing
Reproducability and the correct functioning of code are essential to avoid wasted time. If a code block is copied more
than once then it should be placed into a common script / module under src and unit tests added. The same applies for
any other non trivial code to ensure the correct functioning.

To run tests, install pytest using pip or conda (should have been setup already if you used the conda_env.yml file) and
then from the repository root run

```
pytest
```

## Automated Document Generation
A [sphinx](https://www.sphinx-doc.org/) project is provided under docs/writeup that will generate writeup that
also includes automatically generated API information for any packages. THe output can be created in multiple
formats including html and pdf. If you are using CI then this can be run automatically. To run
locally execute the following commands:

```
cd docs/writeup
make html
```

On Windows this will run the make.bat, a Makefile is also included for those using the 'make' command.

## Development Process
Contributions to this template are greatly appreciated and encouraged.

To contribute an update simply:
* Create a new branch / fork for your updates.
* Check that your code follows the PEP8 guidelines (line lengths up to 120 are ok) and other general conventions within this document.
* Ensure that as far as possible there are unit tests covering the functionality of any new code.
* Check that all existing unit tests still pass.
* Edit this document if needed to describe new files or other important information.
* Create a pull request.

## Important Links
* https://wiki.equinor.com/wiki/index.php/Statoil_Data_Science_Technical_Standards - Data Science Technical Standards (Equinor Internal)
* https://dataplatformwiki.azurewebsites.net/doku.php - Data Platform wiki (Equinor internal)
* https://github.com/equinor/data-science-shared - Shared Data Science Code Repository (Equinor internal)

## References
* https://github.com/equinor/data-science-template/ - The master template for this project
* http://docs.python-guide.org/en/latest/writing/structure/
* https://github.com/Azure/Microsoft-TDSP
* https://drivendata.github.io/cookiecutter-data-science/

[//]: #
   [anaconda]: <https://www.continuum.io/downloads> -->
