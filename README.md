# Project-ML
> Submitted by:
  > Efrat Cohen - 207783150 | 
  > Liel Serfaty - 312350622
  
## Usage
In the `main.py` you can find the variable named `args` that pick the database to run:
```
if __name__ == "__main__":
    # change it for switching databases
    args = 'all'
```

This variable can set to 3 values:
1. int - the **index** of the database (see below)
2. string - the **name** of the database
3. string - 'all' - for run all the databases

## Databases
List of databases:
```
1. 'CNS'
2. 'Lymphoma'
3. 'MLL'
4. 'Ovarian'
5. 'SRBCT'
6. 'ayeastCC'
7. 'bladderbatch'
8. 'CLL'
9. 'ALL'
10. 'leukemiasEset'
11. 'GDS4824'
12. 'khan_train'
13. 'ProstateCancer'
14. 'Nutt-2003-v2_BrainCancer.xlsx - Sayfa1'
15. 'Risinger_Endometrial Cancer.xlsx - Sayfa1'
16. 'madelon'
17. 'ORL'
18. 'Carcinom'
19. 'USPS'
20. 'Yale'
```
## Run
After selecting a database, the file will be loaded, a pipeline of:
* preprocessing will be performed
* selection of features using 7 algorithms (4 for a baseline, 2 that we implemented, and 1 which is an improvement of one of the implementations)
* measuring the performance using the construction of 5 classifiers
* evaluating them using 4 metrics with cross-validation that best fits the data

## Output
For each database the output is csv file contain the metrics for evaluate the ML methods, time for reducing and for build the classifier and predict with its. All that files placed in `output` folder. 
Also, the database after the preprocessing stage will be placed in csv format at `data_process` folder.
