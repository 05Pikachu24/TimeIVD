#  TimeIVD: Self-Supervised Inter-Variable Dependency Transformers for Time Series Forecasting
WSDM 2027

## Prerequisites

Ensure you are using Python >= 3.8 and install the necessary dependencies in the requirements.txt

## Prepare Datastes

Begin by downloading the required datasets. All datasets are conveniently available at [URL](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link). We thank all iTransformer authors who provide the dataset URL. Create a separate folder named `./data` and neatly organize all the csv files as shown below:
```
data
└── PEMS
└── electricity.csv
└── ETTh1.csv
└── ETTh2.csv
└── ETTm1.csv
└── ETTm2.csv
└── exchange_rate.csv
└── solar_AL.txt
└── traffic.csv
└── weather.csv
```

## Training

All scripts are located in `./scripts`. For instance, to train or test a model using the ETTh1 dataset with an input length of 96, simply run:

```shell
bash ./scripts/pretrain/ETTh1.sh

bash ./scripts/finetune/ETTh1.sh
```

After training:

- Your trained model will be safely stored in `./outputs/pretrain_checkpoints` and `./outputs/checkpoints`.

## Verifying the effectiveness of TwoSIVD and ViFilter

iTransformer, CASA: The datasets and storage paths are kept consistent with those used in our TimeIVD (see the description above).
- Run the shell scripts from ./iTransformer/scripts/multivariate_forecasting_TwoSIVD  ./CASA-master/scripts/long_term_forecast_ViFilter/
```shell
# run original iTransformer
bash ./iTransformer/scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh
# run iTransformer integrated our ViFilter 
bash ./iTransformer/scripts/multivariate_forecasting_ViFilter/ETT/iTransformer_ETTh1.sh
# run original CASA 
bash ./CASA-master/scripts/long_term_forecast/ETT_script/CASA_ETTh1.sh
# run CASA integrated our ViFilter 
bash ./CASA-master/scripts/long_term_forecast_ViFilter/ETT_script/CASA_ETTh1.sh

# run iTransformer integrated our TwoSIVD 
bash ./iTransformer/scripts/multivariate_forecasting_TwoSIVD/ETT/iTransformer_ETTh1.sh
```

DUET:
Download the dataset from the well pre-processed datasets from their provided [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view) or [Baidu Drive](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry). Then place the downloaded data under the folder ./DUET/dataset.
- Run the shell scripts from DUET-main/scripts/multivariate_forecast/
```shell
# run the original DUET
bash ./DUET/scripts/multivariate_forecast/DUET.sh
# run the DUET integrated our TwoSIVD
bash ./DUET-main/scripts/multivariate_forecast/DUET_TwoSIVD_ETTh1.sh
```

## Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:


- [iTransformer](https://github.com/thuml/iTransformer)
- [DUET](https://github.com/decisionintelligence/DUET)
- [CASA](https://github.com/lmh9507/CASA)


