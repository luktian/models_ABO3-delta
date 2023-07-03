from PSPABO3 import abo3searching, columns as dcolumns
import joblib, os, numpy as np
from fml.data import read_data

targted_value = -1.35
criterion = 0.1
site_counts = [2, 2]
rounds = 2000

vr = joblib.load("vr.joblib")
output_file = f"outputfile_{targted_value}.csv"

formular_info = [
    ['Yb','Ca','Y','Sr','Ba','Sm','Pr','Nd','La','Cd'],
    ['Al','Ti','Sc','Y','Zn','Mg','Ga','Be','Lu','In','Zr','Co','Ni','Fe']
    ]

new_flag = False
if os.path.exists(output_file):
    try:
        existed_data = read_data(output_file)
        existed_formulas = existed_data.index.value.tolist()
    except:
        new_flag = True
else: 
    new_flag = True
if new_flag:
    existed_formulas = []
    with open(output_file, "w") as f:
        write_str = f"formula, pred, error, {', '.join(dcolumns)}\n"
        f.writelines(write_str)

for i in range(999999):
    
    # rounds = np.random.choice([200, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000])
    
    trials = abo3searching(vr, formular_info, targted_value=targted_value, criterion=criterion, site_counts=site_counts, 
                           ratio_digit=3, rounds=rounds, verbose=False, verbose2=True, verbose3=False)
    for trial in trials:
        formula = trial[0]
        if formula not in existed_formulas:
            existed_formulas.append(formula)
            with open(output_file, "a") as f:
                write_str = f"{formula}, {trial[1]}, {trial[2]}, {', '.join([ str(i) for i in trial[3]])}\n"
                f.writelines(write_str)