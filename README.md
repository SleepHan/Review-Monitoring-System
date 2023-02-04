# Fake-Product-Reviews

## Description
Currently supports Shopee review extraction

Produces CSV file on review
- Username
- Rating
- Comment
- Images
- Videos

## Dependencies
- &gt;= Python 3.7
- pandas

## Usage
```
python reviewExtractShopee.py <Product URL>
```



# Review Monitoring System

## Dependencies
- Install requirements.txt
- Note that some edits have been made to the sgnlp libraries
	- Replace preprocess.py and postprocess.py in "venv/Lib/site-packages/sgnlp/models/sentic_gcn/" with the scripts provided in this repository


## Usage
- Run reviewExtractShopee.py to get the data in csv
- Run main.py --csv <csv file>