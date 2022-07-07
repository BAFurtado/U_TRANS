# Modelling Urban Transition with Coupled Housing and Labour Markets (U-TRANS)

## blind for peer review
 
## blind for peer review 

### Data input to run graphs available in folder: output/
### To generate graphs, run Urban_Transition_1.ipynb

# Model Steps

1. Read any shapefile [current shapefile is set at read_data.py] that contains one column 'Name' and one 'geometry'
2. Model is initialized. 
    1. Industry is based only on parameters (size, workers skill, income (mu, sigma))
    2. Business need an industry they belong to 'businessTy' and a 'geometry' location of type POINT

### Requirements

A conda environment with the following command will do:

```conda create --name p38geo python=3.8 gdal fiona geopandas shapely numpy joblib matplotlib scikit-learn descartes --channel=conda-forge --strict-channel-priority```

May need to run `conda clean -a` first to clean the compressed .bz files and delete older versions of package folders

To activate the environment use:
`source activate p38geo` 

To deactivate
`conda deactivate `

### To run the model

The general main.py procedure works as follows, with a '/' meaning one of the alternatives:

    # STANDARD procedure:
    # <python   main.py
    #           NUMBER_CPUS/plotting
    #           NUMBER_RUNS/names of plotting folders
    #           SENSITIVITY/SCENARIOS/RUN 
    #           ALL/SPECIFIC_SCENARIO/PARAM VALUES>


1. Adjust parameters at "params"
2. Run a single run

```angular2html
python main.py
```

3. You can also establish number of cpus and runs


```angular2html
python main.py 4 10
```
or
```
python main.py 4 10 run
```

4. For scenarios, enter the word 'scenarios' plus the 
**number of cpus and runs** for each one and ALL, or leave it blanc

```angular2html
python main.py 4 10 scenarios ALL
```

5. For a specicif scenario, name it
```angular2html
python main.py 4 10 scenarios TRANSITION
```

6. For sensitivity analysis of parameters run RUNNER.PY adding number of cpus and number of runs 
You may adjust parameters and values at runner.py 'parameters_to_test'
```angular2html
python runner.py 4 10
```

7. Alternatively, you can just name a parameter and the intervals, as such:

Don't use SPACES
```angular2html
python main.py 4 10 sensitivity TRANSITION N_PEOPLE [100,1000,10000]
```

8. For just the plots of sensitivity analysis, run:
```angular2html
python runner.py plotting
```

9. For both data and plots, run:
```angular2html
python runner.py 4 10 plotting
```

10. For one parameter, you may change the parameter "p" and run
```angular2html
python main.py sensitivity 4 10
```

11. To redo the plotting of a previously run scenario, you may run:
```angular2html
python main.py plotting
```