# MolCLR-NAS: Molecular Contrastive Learning for NAS (Neural Architecture Search)

This repository combines the MolCLR model with the AutoGL (Automated Graph Learning) framework to implement a Neural Architecture Search (NAS) algorithm for MolCLR. MolCLR is a molecular contrastive learning framework for molecular representation learning based on Graph Neural Networks (GNNs). It was introduced in the paper ["Molecular Contrastive Learning of Representations via Graph Neural Networks"](https://www.nature.com/articles/s42256-022-00447-x). MolCLR pre-training significantly improves the performance of GNN models on various downstream molecular property prediction benchmarks.

## Prerequisites

Before running the code, please make sure to set up the environment and download the required datasets:

1. Create a new Conda environment:

```shell
	conda create --name molclr python=3.7
	conda activate molclr
```

2. Install the necessary packages:

   ```shell
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
   pip install PyYAML
   conda install -c conda-forge rdkit=2020.09.1.0
   conda install -c conda-forge tensorboard
   conda install -c conda-forge nvidia-apex  # Optional for mixed-precision training
   ```

3. Download the required datasets from the following link:
   [MolCLR NAS Datasets](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view)

## Running MolCLR NAS

Once the environment is set up and the datasets are downloaded, you can run the MolCLR NAS code by executing the `molclr_NAS.py` script.

```shell
python molclr_NAS.py
```

## Configuring NAS Search

In the code, we define a `NAS_top` class that controls the overall NAS search process. You can modify the dataset path and adjust parameters within the `NAS_top` class to switch between different datasets and customize the NAS search process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
