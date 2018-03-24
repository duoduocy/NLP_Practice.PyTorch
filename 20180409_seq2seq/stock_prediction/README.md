# Stock manager

## Software environment
Python packages
```bash
BeautifulSoup==3.2.1
numpy==1.13.1
pandas==0.16.2
scikit-learn==0.16.1
scipy==0.19.1
tensorflow==1.2.1
urllib3==1.8
```

## Stoack data
1. You can download the full S&P 500 data from [Yahoo! Finance ^GSPC](https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC) (click the "Historical Data" tab and select the max time period). And save the .csv file to `data/SP500.csv`.

## Traing and testing

### Data fetcher
Run `python data_fetcher.py` to download the prices of individual stocks in S & P 500, each saved to `data/{{stock_abbreviation}}.csv`.

### Training
Run `python main.py --help` to check the available command line args.
Run `python main.py` to train the model.

For examples,
- Train a model only on SP500.csv; no embedding
```
python main.py --stock_symbol=SP500 --train --input_size=1 --lstm_size=128 --max_epoch=50
```

- Train a model on 100 stocks; with embedding of size 8
```
python main.py --stock_count=100 --train --input_size=1 --lstm_size=128 --max_epoch=50 --embed_size=8
```

## Reference
- [LSTM Neural Network for Time Series Prediction](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)
- [Google Alphabet Inc CSV data](http://finance.google.com/finance?q=nasdaq:goog)
