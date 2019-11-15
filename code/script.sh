# activate python enviroment
source .venv/bin/activate

# Initate them
nohup python3 tcc_1.py > tcc_1.out 2> tcc_1.err &

nohup python3 tcc_2.py > tcc_2.out 2> tcc_2.err &

nohup python3 tcc_3.py > tcc_3.out 2> tcc_3.err &

nohup python3 tcc_4.py > tcc_4.out 2> tcc_4.err &
