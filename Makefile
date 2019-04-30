install: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || python3 -m venv venv
	venv/bin/pip install -Ur requirements.txt
	venv/bin/pip install -e .
	touch venv/bin/activate

train: install
	python3 lsmhun/train_model.py

test: install
	venv/bin/pytest --cov=lsmhun --cov-report xml

pylint: install
	venv/bin/pylint --rcfile=.pylintrc lsmhun -r n > pylint-report.txt

sonar: install
	/opt/sonar-scanner/bin/sonar-scanner
