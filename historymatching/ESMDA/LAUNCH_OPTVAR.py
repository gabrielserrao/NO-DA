import subprocess

# Number of cases you want to run
total_cases = 100

# Path to your main script
script_path = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/VARIATIONAL_FNO_v1.py'

# Iterate over each case number and call the main script
for case in range(total_cases):
    command = f"python {script_path} {case}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"An error occurred while processing case {case}: {result.stderr.decode()}")
    else:
        print(f"Successfully processed case {case}")
