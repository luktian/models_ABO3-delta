
import pandas as pd, numpy as np, re, copy

root_path = __file__.split("descriptor.py")[0]

dm = descriptor_mapping = pd.read_excel(root_path+"common.xlsx", index_col=0).astype(float)
rm = radii_mapping = pd.read_excel(root_path+"radii.xlsx", index_col=0).astype(float)

columns = ["Za", "A_Tm", "A_Tb", "A_Hfus", "Radius_A"] + ["Zb", "B_Tm", "B_Tb", "B_Hfus", "Radius_B"] + ["R_a/R_b"] + ["TF", "aO3"]

def split_parts_by_compatiability(_formular, ratios):
    formular = copy.deepcopy(_formular)
    formular_list = list()
    for r in ratios:
        site = re.split(r, formular)[0] + r
        formular_list.append(site)
        site_len = len(site)
        formular = formular[site_len:]
    return formular_list

def _sortdict(adict,reverse=False):
    keys = list(adict.keys())
    keys.sort(reverse=reverse)
    return {key:adict[key] for key in keys}

def split_formula(formula):
    ratios = re.compile("[\d+\.]+").findall(formula)
    splited_parts_by_ratios = re.split("|".join(ratios), formula)
    splited_parts_by_ratios = split_parts_by_compatiability(formula, ratios)
    parts_with_ratios_list = []
    for ratio, part in zip(ratios+["1"], splited_parts_by_ratios):
        
        if len(part) == 0 or part == " ":
            continue
        
        subparts = re.compile("[A-Z]{1,2}(?![a-z])|[A-Z]{1}[a-z]{1,2}|[A-Z]{3}(?![a-z])|(?<=[N])[WVYP]").findall(part)
        
        if len(subparts) > 1:
            for i,j in enumerate(subparts[:-1]):
                subparts[i] = subparts[i] + "1"
        # print(part)
        subparts[-1] = subparts[-1] + ratio
        
        parts_with_ratios_list += subparts
    
    ratio_sums = np.array([1, 1, 3]).astype(float)
    site_list = [ {} for i in range(len(ratio_sums)) ]
    
    formular_list = parts_with_ratios_list
    
    formular_i = 0
    site_i = 0
    ratios = 0
    while formular_i < len(formular_list):
        _formular = formular_list[formular_i]
        ratio_sum = ratio_sums[site_i]
        ratio = re.compile("[\d+\.]+").findall(_formular)[0]
        ele = _formular.split(str(ratio))[0]
        ratios += float(ratio)
        site_list[site_i].update({ele:ratio})
        
        if round(ratios, 10) == round(ratio_sum, 10):
            site_i += 1
            ratios = 0
        formular_i += 1
    return [ _sortdict(i) for i in site_list ]

def split_formulas(formulas):
    splitted_formulas = []
    for formula in formulas:
        splitted_formulas.append(split_formula(formula))
    return splitted_formulas

def generate_abo3(formula):
    """

    formula: [{"Li": 1}, {"Bi": 0.1, "Ca": 0.9}]

    """

    descriptors = []
    radius = []
    for site in formula:
        site_descriptor = []
        for atom, ratio in site.items():
            descriptor_atom = []
            # common descriptors
            descriptor_atom += (dm.loc[atom] * float(ratio)).values.tolist()
            # radii
            radii = rm.loc[atom]
            for i in [3, 2, 1]:
                if str(radii[i]) == "nan":
                    continue
                else:
                    radii = radii[i] * float(ratio)
                    break
            descriptor_atom += [radii]
            site_descriptor.append(np.array(descriptor_atom))
        site_descriptor = np.sum(site_descriptor, axis=0)
        radius.append(site_descriptor[-1])
        descriptors += site_descriptor.tolist()
    
    ra = radius[0]
    rb = radius[1]
    
    ra_div_rb = ra / rb
    
    tf = (ra + 140.0) / (np.sqrt(2) * (rb + 140.0))
    
    aO3 = 2.37 * rb + 2.47 - 2 * ((1 / tf) - 1)
    
    descriptors += [ra_div_rb, tf, aO3]
    
    return np.array(descriptors).reshape(1, -1)


if __name__ == '__main__':
    
    formulas = pd.read_csv("../dataset.csv").iloc[:, 0]
    
    formulas = formulas.values.tolist()
    splitted_formulas = split_formulas(formulas)
    
    # formula = [
    #     {"Sr": 0.9, "Na": 0.1},
    #     {"Sc": 0.5, "Al": 0.5}
    # ]

    descriptors = []
    for formula in splitted_formulas:
        a = generate_abo3(formula[:-1])
        descriptors.append(a)

    descriptors = np.concatenate(descriptors, axis=0)
    descriptors = pd.DataFrame(descriptors, columns=columns, index=formulas)


