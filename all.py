import subprocess
import pprint

agents = {
    'TabularQR': 'TR',
    'TabularQQ': 'TT',
    'TabularMR': 'SR',
    'TabularMM': 'SS',
    'QR': 'QR',
}

runs = ['TabularQR', 'TabularQQ', 'TabularMR', 'TabularMM']

results = {}
for run in runs:
    # Clear previous tensorboard results
    subprocess.run(['rm', '-rf', 'logs/' + run])

    # Train, eval and challenge
    output = subprocess.run(['python', 'run_dqn_deepsoccer.py',
                    '--agents=' + agents[run], '--eval', '--challenge',
                    '--batch=1', '--starts=0', '--replay=2', '--freq=1',
                    '--timesteps=400000', '--name=' + run], stdout=subprocess.PIPE)
    raw = output.stdout.decode('utf-8')
    result = list(filter(lambda line: 'vs' in line, raw.split('\n')))
    results[run] = result
    pprint.pprint(results)
