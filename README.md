# Review Monitoring System

## Description
The Review Monitoring System is a project created for the National AI Student Challenge 2022.

The purpose of this project is to combat the issue of fake reviews. Our solution aims to detect reviews that are misleading, either through intentional manipulation by the seller or by users who provide quick, unhelpful ratings for the sake of receiving rewards.


## Installation
- Install Python (>= 3.9)
- Install requirements.txt
```
pip install -r requirements.txt
```
- Overwrite sgnlp libary's `preprocess.py` and `postprocess.py` scripts 
	- Scripts to be replaces can found in
		- Global: `<python path>/Lib/site-packages/sgnlp/models/sentic_gcn/`
		- Virtual Environment: `<venv path>/Lib/site-packages/sgnlp/models/sentic_gcn/`
	- Replace them with `preprocess.py` and `postprocess.py` scripts found in this repository


## Run
1. Prepare CSV file or URL to product page
	- CSV files will require `comment` and `rating` headers
	- Only Shopee URLs are supported 

2. Run `main.py` script
```
py main.py (--csv <CSV File Path> | --url (URL to product))
```
