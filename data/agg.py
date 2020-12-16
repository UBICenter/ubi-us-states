import pandas as pd
import numpy as np
import microdf as mdf


person = pd.read_csv("asec1719.csv.gz")

# Lower column headers
person.columns = person.columns.str.lower()

person.adjginc.replace({99999999: 0}, inplace=True)
person.asecwt /= 3  # Since there are three years.

person["age_group"] = np.where(person.age > 17, "adult", "child")

spmu = person.groupby(["spmfamunit", "year"])[["adjginc"]].sum()
spmu.columns = ["spmu_agi"]
person = person.merge(spmu, left_on=["spmfamunit", "year"], right_index=True)

person["weighted_agi"] = person.asecwt * person.adjginc

state = person.groupby(["statefip"])[["weighted_agi", "asecwt"]].sum()
state.columns = ["state_total_agi", "state_population"]
person = person.merge(state, left_on=["statefip"], right_index=True)

person["state_per_dollar_tax_rate"] = (
    person.state_population / person.state_total_agi
)

population = person.asecwt.sum()
total_agi = (person.adjginc * person.asecwt).sum()
fed_tax_per_dollar = population / total_agi


def pov(df, ubi):
    """Calculate poverty rate across a set of person records and UBI amount.

    Args:
        df: DataFrame with records for each person.
        ubi: Annual UBI amount.

    Return:
        SPM poverty rate.
    """
    # Calculate required tax rate.
    tax_rate = fed_tax_per_dollar * ubi
    # Add UBI, subtract new tax liability.
    new_spmtotres = df.spmtotres + (ubi * df.pernum) - (df.spmu_agi * tax_rate)
    # Recalculate SPM poverty flags.
    new_pov = new_spmtotres < df.spmthresh
    # Return weighted average of the SPM flags (poverty rate).
    result = (new_pov * df.asecwt).sum() / df.asecwt.sum()
    return pd.Series({"poverty_rate": result})


age = person.groupby(["age_group"]).apply(lambda x: pov(x, 1000))

age_state = person.groupby(["age_group", "statefip"]).apply(
    lambda x: pov(x, 1000)
)

UBI_AMOUNTS = np.arange(0, 12001, 1000)


def pov_groupby(groupby, ubi_amounts=UBI_AMOUNTS):
    l = []

    for i in UBI_AMOUNTS:
        tmp_age = pd.DataFrame(
            person.groupby(groupby).apply(lambda x: pov(x, i))
        )
        tmp_age["ubi"] = i
        l.append(tmp_age)

    return pd.concat(l)


pov_groupby("age_group").to_csv("age_all.csv")
pov_groupby(["age_group", "statefip"]).to_csv("age_state_all.csv")
