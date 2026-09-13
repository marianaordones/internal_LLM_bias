# Occupation cues for the inferred-income experiment

The occupation cues were selected from three 2025 Bureau of Labor Statistics
(BLS) tables:

- OEWS national estimates: median annual wage by detailed occupation;
- Employment Projections Table 5.4: typical education needed for entry;
- CPS Annual Averages Table 11: percentage of employed workers who are women.

## Selection rule

The selection uses detailed occupations that satisfy all of the following:

1. the typical entry-level education is exactly `High school diploma or equivalent`;
2. the OEWS median annual wage is available;
3. the occupation title matches exactly, after punctuation and case normalization,
   between the education, OEWS, and CPS tables;
4. women account for 35% to 65% of workers in CPS Table 11.

Using one exact education category, instead of mixing `high school` and `no formal
credential`, keeps the BLS education assignment constant across both income
groups. The gender-composition interval removes occupations with a strongly
gender-skewed workforce. The filters leave 27 eligible occupations. The low-income
group is the ten lowest-wage occupations and the high-income group is the ten
highest-wage occupations among those eligible; the seven middle occupations are
not used.

The resulting mean annual wages are $42,557 for the low-income group and $63,295
for the high-income group. Their wage ranges do not overlap ($38,120--$46,380 and
$53,830--$69,990, respectively). Mean female representation is 53.3% and 49.4%,
respectively.

`occupation_income_cues.csv` contains the selected cues and their provenance.
`occupation_income_selection_candidates.csv` contains all eligible occupations,
their wage ranks, and the selection decision.

## Interpretation and limitations

The BLS education variable is an occupation-level assignment describing the
typical education needed for entry. It is not the observed educational attainment
of every worker. Related work experience also varies across occupations, especially
for some managerial occupations in the high-income tail. Therefore, the experiment
should test all demographic probes after the cue. A response effect should not be
attributed specifically to socioeconomic status if the cue also produces a material
education, age, or gender-probe shift.

The prompt occupation names are singular, minimally edited versions of the official
BLS titles. The official title and SOC code remain in the CSV for auditing.

## Sources

- https://www.bls.gov/oes/tables.htm
- https://www.bls.gov/emp/tables/education-and-training-by-occupation.htm
- https://www.bls.gov/cps/cpsaat11.htm
- https://www.bls.gov/emp/documentation/education/tech.htm
