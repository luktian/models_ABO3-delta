
from .descriptor import generate_abo3, columns as dcolumns
from hyperopt import fmin, hp, STATUS_OK, tpe
from fml.data import DataObject

def abo3searching(votingmodel, formular_info, targted_value, criterion, site_counts=[2, 2, 1], ratio_digit=3, rounds=100, verbose=False, verbose2=False, verbose3=False):
    
    """
    formular_info = [
        ["FA", "MA", ...] # atoms in A site
        ["Pb", "Sn", ...] # atoms in B site
    ]
    site_counts = [2, 2]
    """
    
    formulas = []
    trials = []
    
    space = {}
    for site, site_info, site_count in zip(["A", "B"], formular_info, site_counts):
        atom_info = {}
        ratio_info = {}
        max_ratio = 1
        for site_i in range(1, site_count+1):
            atom_name = f"{site.lower()}{site_i}"
            ratio_name = f"r{atom_name}"
            atom_info.update({
                atom_name: hp.choice(atom_name, site_info)
                # atom_name: [atom_name, site_info]
            })
            ratio_info.update({
                ratio_name: hp.uniform(ratio_name, 0, max_ratio)
                # ratio_name: [ratio_name, 0, max_ratio]
            })
        space[site] = {
            "atoms": atom_info,
            "ratios": ratio_info,
        }
    
    def f(params):
        if verbose:
            print(params)
        formular = []
        formular_name = ""
        for _, site_info in params.items():
            tmp = {}
            ratio_sum = sum(site_info["ratios"].values())
            for atom, ratio in zip(site_info["atoms"].values(), site_info["ratios"].values()):
                ratio = ratio / ratio_sum
                if _ == "C":
                    ratio *= 3
                ratio = round(ratio, ratio_digit)
                if atom in tmp.keys():
                    tmp[atom] += ratio
                else:
                    tmp[atom] = ratio
            for atom, ratio in tmp.items():
                formular_name += atom
                if ratio != 1:
                    formular_name += str(round(ratio, ratio_digit))
            formular.append(tmp)
        descriptor = generate_abo3(formular)
        data = DataObject(X=descriptor, Y=[0], 
                          Xnames=dcolumns, Yname=votingmodel.trainobjects[0].Yname[0])
        pred = votingmodel.predict(data)[0]
        if verbose3:
            print(pred)
        error = abs(pred - targted_value)
        if error < criterion:
            if formular_name not in formulas:
                formulas.append(formular_name)
                trials.append([formular_name, pred, error] + descriptor.tolist())
                if verbose2:
                    print(f"{formular_name}: {pred}")
        return {"loss": error, "status": STATUS_OK}
    fmin(fn=f, space=space, algo=tpe.suggest, max_evals=rounds, verbose=verbose2)
    return trials


if __name__ == "__main__":
    
    import joblib
    vr = joblib.load("../vr.joblib")
    
    columns = []
    for i in vr.trainobjects:
        columns += i.Xnames.tolist()
    columns = list(set(columns))
    
    formular_info = [
        ['Yb','Ca','Y','Sr','Ba','Sm','Pr','Nd','La','Cd'],
        ['Al','Ti','Sc','Y','Zn','Mg','Ga','Be','Lu','In','Zr','Co','Ni','Fe']
        ]
    
    trials = abo3searching(vr, formular_info, targted_value=0, criterion=0.05, site_counts=[2, 2], 
                           ratio_digit=3, rounds=1000, verbose=False, verbose2=True, verbose3=True)
    
    