import subprocess
import pprint

agents = {
    'TabularQR': 'TR',
    'TabularQQ': 'TT',
    'TabularMR': 'SR',
    'TabularMM': 'SS',
    'DeepQR': 'QR',
    'DeepMR': 'MR',
}

runs = ['DeepQR']

results = {}
for run in runs:
    # Clear previous tensorboard results
    subprocess.run(['rm', '-rf', 'logs/' + run])

    # Train, eval and challenge
    output = subprocess.run(['python', 'run_dqn_deepsoccer.py',
                             '--agents=' +
                             agents[run], '--eval', '--challenge',
                             '--batch=32', '--starts=50000', '--replay=1000000', '--freq=4',
                             '--timesteps=10000000', '--name=' + run], stdout=subprocess.PIPE)
    raw = output.stdout.decode('utf-8')
    result = list(filter(lambda line: 'vs' in line, raw.split('\n')))
    results[run] = result
    pprint.pprint(results)
