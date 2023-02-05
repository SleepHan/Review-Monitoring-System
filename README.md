# Review Monitoring System

## Description
The Review Monitoring System is a project created for the National AI Student Challenge 2022.

The purpose of this project is to combat the issue of fake reviews. Our solution aims to detect reviews that are misleading, either through intentional manipulation by the seller or by users who provide quick, unhelpful ratings for the sake of receiving rewards.


## Limitations
As of right now, our project will only work for Shopee products


## Installation
- Install Python (>= 3.9)
- Install requirements.txt
```
pip install -r requirements.txt
```
- Overwrite sgnlp libary's `preprocess.py` and `postprocess.py` scripts 
	- Scripts to be replaces can found in `venv/Lib/site-packages/sgnlp/models/sentic_gcn/`
	- Replace them with `preprocess.py` and `postprocess.py` scripts found in this repository


## Run
1. Prepare CSV file or URL to Shopee prodcut page
	- Note that the CSV file prepared will require `comment` and `rating` headers

2. Run `main.py` script
```
py main.py (--csv <CSV File Path> | --url (URL to product))
```
