# scGNN × HPO

This project is an extension of the **scGNN** model (https://github.com/juexinwang/scGNN), incorporating an **HPO** (Hyperparameter Optimization) search component using our team's **AutoGL** framework. The aim is to explore optimal parameter combinations within the original model using HPO techniques. 

To run this example, you should first follow the dataset installation instructions provided in scGNN-README.md. Additionally, ensure that you have successfully installed the AutoGL framework, and consult the scGNN-requirements.txt file for installing the necessary pip packages.

Once you have set up the dataset and environment, you can proceed by following the instructions in scGNN-README.md and execute the following command to initiate the hyperparameter search:
```bash
python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 10 --EM-epochs 20 --quickmode --nonsparseMode
```

In comparison to the original scGNN model files, this example primarily involves modifications to the **scGNN.py** and **model.py** files to adapt the model for HPO tasks. Currently, the model focuses on hyperparameter search related to model width, depth, and optimizers. You can make modifications to the `do_hpo()` function within the scGNN.py file and the `scGNN` to explore other hyperparameters of interest.