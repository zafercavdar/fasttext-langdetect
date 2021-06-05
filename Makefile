publish:
	python3 setup.py sdist && twine upload dist/*

test:
	python3 -m unittest tests/test_detect.py
