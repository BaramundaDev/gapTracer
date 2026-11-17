.PHONY: install test demo
install:
	pip install -e ".[dev,serve]"
test:
	pytest -q
demo:
	gaptrace demo --data data && gaptrace rebuild --data data && gaptrace gaps --data data
