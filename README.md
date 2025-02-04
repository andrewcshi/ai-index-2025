# 2025 AI Index Report

This repository contains the code used to scrape {AAAI, AIES, FAccT, ICML, ICLR, NeurIPS} papers from their respective conference websites.

## To run main conference scrapers:

First, clone the repository:

```
git clone https://github.com/andrewcshi/ai-index-2025.git
cd ai-index-2025
```

Then, install the requirements:

```
pip install -r requirements.txt
```

Then, run the scrapers using the following commands:

```
python main/aaai.py
python main/aies.py
python main/facct.py
python main/icml.py
python main/iclr.py
python main/neurips.py
```

To print statistics for the papers, run the following command:

```
python data/stats.py
```