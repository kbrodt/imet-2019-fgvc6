# iMet Collection 2019 - FGVC6

This is an another one approach to solve the competition from kaggle
[iMet Collection 2019 - FGVC6](https://www.kaggle.com/c/imet-2019-fgvc6).

47th place over 446 (silver medal) with 0.636 F2 score (top 1 -- 0.672).

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

First download the train and test data from the competition link and put them
into [./data](./data) folder.

To train the model run

```python
python ./src/train.py
```

As this competition is kernel only, this command generates only trained models.
It also needed to do inference on best chechpoints and averege the results via simple mean.

### Approach

Detailed solution see in
[this presentation](presentation/iMet_Collection_2019-FGVC6.pdf).
