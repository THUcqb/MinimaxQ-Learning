tabularQR:
	rm -rf logs/TabularQR
	python run_dqn_deepsoccer.py --agents=TR --eval --challenge --batch=1 --starts=0 --replay=1 --freq=1 --name=TabularQR

tabularQQ:
	rm -rf logs/TabularQQ
	python run_dqn_deepsoccer.py --agents=TT --eval --challenge --batch=1 --starts=0 --replay=1 --freq=1 --name=TabularQQ

tabularMR:
	rm -rf logs/TabularMR
	python run_dqn_deepsoccer.py --agents=SR --eval --challenge --batch=1 --starts=0 --replay=1 --freq=1 --name=TabularMR

tabularMM:
	rm -rf logs/TabularMM
	python run_dqn_deepsoccer.py --agents=SS --eval --challenge --batch=1 --starts=0 --replay=1 --freq=1 --name=TabularMM

