publish:
	python3 setup.py sdist && twine upload dist/*

test:
	pytest
