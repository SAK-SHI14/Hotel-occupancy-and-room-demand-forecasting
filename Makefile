.PHONY: setup data clean-data features train evaluate test api app all check

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m hotel_forecast.data.generate
	$(PYTHON) -m hotel_forecast.data.clean
	$(PYTHON) -m hotel_forecast.data.validate

features:
	$(PYTHON) -m hotel_forecast.features.build

train:
	$(PYTHON) -m hotel_forecast.models.train

evaluate:
	$(PYTHON) -m hotel_forecast.evaluate --split test

test:
	$(PYTHON) -m pytest -v

api:
	$(PYTHON) -m uvicorn hotel_forecast.api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/dashboard.py

check:
	ruff check .
	$(PYTHON) -m pytest -q

all: data features train evaluate test
