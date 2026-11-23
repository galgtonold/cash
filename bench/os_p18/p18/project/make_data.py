"""Generate a synthetic clinical-trial data transfer (CSV files in data/).

python make_data.py              # transfer 1
python make_data.py --transfer 2 # later transfer: late AEs, updated dispositions, late labs from 3 sites
"""
import argparse
import os

import numpy as np
import pandas as pd

TESTS = {
    # code: (mean, sd, std unit, alt unit, factor alt->std)
    "ALT": (28, 12, "U/L", None, None),
    "AST": (25, 10, "U/L", None, None),
    "GLUC": (5.4, 1.1, "mmol/L", "mg/dL", 1 / 18.016),
    "CREAT": (80, 18, "umol/L", "mg/dL", 88.42),
    "HGB": (138, 14, "g/L", "g/dL", 10.0),
    "WBC": (6.5, 1.8, "10^9/L", None, None),
    "PLT": (250, 60, "10^9/L", None, None),
    "SODIUM": (140, 3, "mmol/L", None, None),
    "POTASSIUM": (4.2, 0.4, "mmol/L", None, None),
    "CHOL": (5.1, 1.0, "mmol/L", "mg/dL", 1 / 38.67),
    "TRIG": (1.5, 0.7, "mmol/L", "mg/dL", 1 / 88.57),
    "HDL": (1.4, 0.35, "mmol/L", "mg/dL", 1 / 38.67),
    "LDL": (3.0, 0.8, "mmol/L", "mg/dL", 1 / 38.67),
    "ALB": (42, 4, "g/L", "g/dL", 10.0),
}
US_LIKE = {"USA", "Canada", "Mexico", "Brazil"}
COUNTRIES = ["USA", "Canada", "Germany", "France", "Spain", "Poland", "UK", "Japan", "Brazil", "Mexico", "Australia", "Italy"]
ARM_SPELLINGS = {
    "Placebo": ["Placebo", "PLACEBO", "PBO", "placebo"],
    "Low dose": ["Low dose", "LOW DOSE", "Drug 10mg", "low"],
    "High dose": ["High dose", "HIGH DOSE", "Drug 40mg", "high"],
}
SEX_SPELLINGS = {"F": ["F", "Female", "f", "FEMALE"], "M": ["M", "Male", "m", "MALE"]}
AE_TERMS = {
    "Headache": ["Headache", "headache", "HEADACHE", "Head ache", "headache ", "Headache (mild)"],
    "Nausea": ["Nausea", "nausea", "NAUSEA", "Nausea ", "feeling sick"],
    "Fatigue": ["Fatigue", "fatigue", "Tiredness", "tired", "FATIGUE"],
    "Dizziness": ["Dizziness", "dizzy", "Dizzyness", "DIZZINESS"],
    "Diarrhoea": ["Diarrhoea", "Diarrhea", "diarrhea", "DIARRHEA", "loose stools"],
    "Rash": ["Rash", "rash", "Skin rash", "RASH", "rash on arm"],
    "Insomnia": ["Insomnia", "insomnia", "Sleeplessness", "trouble sleeping"],
    "ALT increased": ["ALT increased", "Elevated ALT", "ALT elevation", "alt raised"],
    "Hypoglycaemia": ["Hypoglycaemia", "Hypoglycemia", "low blood sugar", "HYPOGLYCEMIA"],
    "Back pain": ["Back pain", "back pain", "Backache", "BACK PAIN", "lower back pain"],
    "Nasopharyngitis": ["Nasopharyngitis", "Common cold", "cold", "URTI", "nasopharyngitis"],
    "Arthralgia": ["Arthralgia", "joint pain", "Joint Pain", "ARTHRALGIA"],
}
SOC = {
    "Headache": "Nervous system", "Dizziness": "Nervous system", "Nausea": "Gastrointestinal",
    "Diarrhoea": "Gastrointestinal", "Fatigue": "General", "Rash": "Skin", "Insomnia": "Psychiatric",
    "ALT increased": "Investigations", "Hypoglycaemia": "Metabolism", "Back pain": "Musculoskeletal",
    "Arthralgia": "Musculoskeletal", "Nasopharyngitis": "Infections",
}
N_SITES = 80
N_SUBJECTS = 50_000
N_VISITS = 10


def pick(rng, mapping, keys):
    out = np.empty(len(keys), dtype=object)
    for k, spellings in mapping.items():
        m = keys == k
        out[m] = rng.choice(spellings, size=m.sum())
    return out


def make_transfer1(rng):
    sites = pd.DataFrame({
        "site_id": [f"S{i:03d}" for i in range(1, N_SITES + 1)],
        "country": rng.choice(COUNTRIES, N_SITES),
        "investigator": [f"Dr. {n}" for n in rng.choice(["Meyer", "Smith", "Garcia", "Tanaka", "Nowak", "Rossi", "Dubois", "Brown", "Silva", "Kim"], N_SITES)],
    })
    site_of = rng.integers(0, N_SITES, N_SUBJECTS)
    arm = rng.choice(list(ARM_SPELLINGS), N_SUBJECTS)
    sex = rng.choice(["F", "M"], N_SUBJECTS)
    rand_date = pd.Timestamp("2024-01-08") + pd.to_timedelta(rng.integers(0, 420, N_SUBJECTS), unit="D")
    birth = rand_date - pd.to_timedelta((rng.normal(52, 11, N_SUBJECTS).clip(18, 85) * 365.25).astype(int), unit="D")
    disp = rng.choice(["Completed", "Ongoing", "Withdrawn: adverse event", "Withdrawn: consent", "Lost to follow-up"],
                      N_SUBJECTS, p=[0.55, 0.3, 0.06, 0.06, 0.03])
    sid = np.array([f"{sites.site_id[s]}-{i:05d}" for i, s in enumerate(site_of, 1)])
    subjects = pd.DataFrame({
        "subject_id": sid, "site_id": sites.site_id.values[site_of],
        "arm": pick(rng, ARM_SPELLINGS, arm), "sex": pick(rng, SEX_SPELLINGS, sex),
        "birth_date": birth.strftime("%Y-%m-%d"), "rand_date": rand_date.strftime("%Y-%m-%d"),
        "weight_kg": rng.normal(80, 15, N_SUBJECTS).round(1), "disposition": disp,
    })
    # a few duplicated subject rows (re-entered at the site)
    dup = subjects.sample(120, random_state=1)
    subjects = pd.concat([subjects, dup]).sort_values("subject_id", kind="stable").reset_index(drop=True)

    # visits: completed subjects have all, others drop out
    n_done = np.where(disp == "Completed", N_VISITS, np.where(disp == "Ongoing", rng.integers(3, N_VISITS + 1, N_SUBJECTS), rng.integers(1, N_VISITS, N_SUBJECTS)))
    subj_rep = np.repeat(np.arange(N_SUBJECTS), n_done)
    visit_no = np.concatenate([np.arange(1, k + 1) for k in n_done])
    vdate = rand_date.values[subj_rep] + ((visit_no - 1) * 28 + rng.integers(-3, 4, len(visit_no))).astype("timedelta64[D]")
    visits = pd.DataFrame({"subject_id": sid[subj_rep], "visit": [f"V{v}" for v in visit_no], "visit_date": pd.to_datetime(vdate).strftime("%Y-%m-%d")})

    # labs: every test at every visit, 3% missing, 1.5% re-tests
    nt = len(TESTS)
    li = np.repeat(np.arange(len(visits)), nt)
    codes = np.tile(np.array(list(TESTS)), len(visits))
    keep = rng.random(len(li)) > 0.03
    li, codes = li[keep], codes[keep]
    subj_idx = subj_rep[li]
    vno = visit_no[li]
    country = sites.country.values[site_of[subj_idx]]
    alt_unit = np.isin(country, list(US_LIKE))
    arm_eff = np.select([arm[subj_idx] == "Low dose", arm[subj_idx] == "High dose"], [0.03, 0.08], 0.0)
    values = np.empty(len(li))
    units = np.empty(len(li), dtype=object)
    for code, (mu, sd, ustd, ualt, fac) in TESTS.items():
        m = codes == code
        trend = 1 - arm_eff[m] * (vno[m] - 1) / (N_VISITS - 1) if code in ("GLUC", "CHOL", "LDL", "TRIG") else 1 + (arm_eff[m] * 0.5 * (vno[m] - 1) / (N_VISITS - 1) if code in ("ALT", "AST") else 0)
        v = np.abs(rng.normal(mu, sd, m.sum()) * trend)
        u = np.full(m.sum(), ustd, dtype=object)
        if ualt is not None:
            a = alt_unit[m]
            v[a] = v[a] / fac
            u[a] = ualt
        values[m] = v
        units[m] = u
    labs = pd.DataFrame({
        "subject_id": sid[subj_idx], "visit": visits.visit.values[li], "test_code": codes,
        "value": values.round(3), "unit": units, "collected": visits.visit_date.values[li],
    })
    retest = labs.sample(frac=0.015, random_state=2)
    retest = retest.assign(value=(retest.value * rng.normal(1, 0.05, len(retest))).round(3))
    labs = pd.concat([labs, retest], ignore_index=True)
    labs.loc[rng.random(len(labs)) < 0.004, "value"] = np.nan

    # vitals per visit
    vitals = pd.DataFrame({
        "subject_id": visits.subject_id, "visit": visits.visit,
        "sbp": rng.normal(128, 15, len(visits)).round(), "dbp": rng.normal(80, 9, len(visits)).round(),
        "hr": rng.normal(72, 10, len(visits)).round(), "weight_kg": (subjects.weight_kg.values[:N_SUBJECTS][subj_rep] if False else rng.normal(80, 15, len(visits))).round(1),
    })
    ae = make_ae(rng, sid, arm, rand_date, n_done, scale=1.0)
    ref = pd.DataFrame([
        {"test_code": c, "unit": TESTS[c][2], "low": round(TESTS[c][0] - 2 * TESTS[c][1], 3), "high": round(TESTS[c][0] + 2 * TESTS[c][1], 3)}
        for c in TESTS
    ])
    return dict(sites=sites, subjects=subjects, visits=visits, labs=labs, vitals=vitals, ae=ae, ref_ranges=ref)


def make_ae(rng, sid, arm, rand_date, n_done, scale):
    rate = np.select([arm == "Low dose", arm == "High dose"], [6.5, 8.0], 5.5) * scale
    n_ae = rng.poisson(rate)
    s = np.repeat(np.arange(len(sid)), n_ae)
    terms = rng.choice(list(AE_TERMS), len(s), p=np.array([14, 10, 10, 7, 7, 6, 5, 4, 3, 9, 15, 10]) / 100)
    onset = rand_date.values[s] + (rng.random(len(s)) * n_done[s] * 28).astype("timedelta64[D]")
    sev = rng.choice(["MILD", "Mild", "1", "MODERATE", "Moderate", "2", "SEVERE", "3"], len(s), p=[0.25, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05])
    return pd.DataFrame({
        "subject_id": sid[s], "ae_verbatim": pick(rng, AE_TERMS, terms),
        "onset_date": pd.to_datetime(onset).strftime("%Y-%m-%d"), "severity": sev,
        "serious": rng.choice(["N", "Y"], len(s), p=[0.96, 0.04]), "related": rng.choice(["Not related", "Possibly", "Probably", "Related"], len(s)),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", type=int, default=1)
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(20240108)
    t = make_transfer1(rng)
    if args.transfer >= 2:
        r2 = np.random.default_rng(2)
        subj = t["subjects"]
        # updated dispositions: some ongoing subjects completed or withdrew
        ongoing = subj.index[subj.disposition == "Ongoing"]
        ch = r2.choice(ongoing, size=len(ongoing) // 4, replace=False)
        subj.loc[ch, "disposition"] = r2.choice(["Completed", "Withdrawn: adverse event", "Withdrawn: consent"], len(ch), p=[0.8, 0.1, 0.1])
        # late-entered AEs
        base = subj.drop_duplicates("subject_id").reset_index(drop=True)
        arm = base.arm.str.lower().map(lambda a: "Placebo" if a in ("placebo", "pbo") else ("Low dose" if a in ("low dose", "drug 10mg", "low") else "High dose")).values
        late = make_ae(r2, base.subject_id.values, arm, pd.to_datetime(base.rand_date), np.full(len(base), 6), scale=0.08)
        t["ae"] = pd.concat([t["ae"], late], ignore_index=True)
        # late lab transfer from three sites: re-tested values appended
        late_sites = ["S007", "S031", "S058"]
        m = t["labs"].subject_id.str[:4].isin(late_sites) & (t["labs"].visit == "V1")
        extra = t["labs"][m].copy()
        extra["value"] = (extra["value"] * r2.normal(1.0, 0.03, len(extra))).round(3)
        t["labs"] = pd.concat([t["labs"], extra], ignore_index=True)
    for name, df in t.items():
        df.to_csv(f"data/{name}.csv", index=False)
        print(f"data/{name}.csv: {len(df):,} rows")


if __name__ == "__main__":
    main()
