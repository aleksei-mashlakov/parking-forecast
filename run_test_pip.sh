echo "Running As: $(whoami):$(id -gn)"
apk update \
&& apk add --no-cache python3=3.6 \
&& apk --no-cache add freetype-dev \
&& apk --no-cache add lapack-dev \
&& apk --no-cache add gfortran \
&& apk --no-cache add python3-dev \

rm -rf testenv
mkdir testenv
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
virtualenv -p python3 ./testenv/test
source ./testenv/test/bin/activate
# pip install tensorflow
python3 -m pip install tensorflow-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl
python3 -m pip install --no-cache-dir -r ./requirements.txt
python3 -m pytest -v --cov ./tests --cov-report term-missing --cov-fail-under=50
#coverage run run_tests.py
#coverage report -m --omit=./testenv/*,run_tests.py  --fail-under=80
#python -m trace --ignore-dir=$(python -c 'import sys ; print ":".join(sys.path)[1:]') -t run_tests.py
deactivate
rm -rf testenv
