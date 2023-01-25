install:
	@pip install -e .

image_run:
	@python indiscover/main.py

text_run:
	@python indiscover/main.py

text_prep:
	@python indiscover/util.py
