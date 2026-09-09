# Societal-good model: comprehensive findings

## Executive summary

The final scoring table contains **29,695,839 unique people**. The mean model probability is **21.92%**, the median is **0.02%**, and **20.90%** receive the model's positive label. The distribution is highly polarized: **65.1%** of probabilities are below 0.001 and **18.7%** are at least 0.99.

The strongest descriptive findings are:

1. **Social-impact fields dominate.** Health majors have a 37.14% positive-label rate and education majors 33.74%, compared with 16.55% for STEM and 17.76% for business.
2. **Employer mission matters even within occupation.** Nonprofit workers score 23.69 percentage points above public-company workers in the raw comparison. Comparing people in the same broad occupation, nonprofit employment remains **10.33 points above private employment** on a weighted basis.
3. **Smaller employers score higher.** Organizations with 1–10 employees are at 27.22%, versus 16.84% at organizations with 10,001+. The within-occupation small-versus-large gap remains **3.84 points**.
4. **Liberal-arts colleges show a modest, consistent advantage.** Liberal-arts colleges score **1.92 points** above other four-year colleges person-weighted and **0.44 points** college-weighted. The advantage is positive in all five selectivity bands and is strongest among selective and moderately selective institutions.
5. **HBCUs stand out.** Matched HBCUs have a 29.39% rate versus 24.27% for matched non-HBCUs, a **5.12-point descriptive gap**.
6. **Gender differences are pervasive within occupation.** Female profiles exceed male profiles in 94 of 96 broad occupations; the median within-occupation gap is **5.73 points**.
7. **Younger cohorts score higher within the same occupations.** People born in 1980–1999 score **4.74 points** above those born in 1950–1969 on a weighted within-occupation basis; 42 of 43 occupations show a positive gap.
8. **The model is highly sensitive to profile-summary length.** Positive-label rates rise from 6.38% for summaries under 100 characters to 38.57% for summaries of 1,000+ characters. This is the largest threat to interpreting the results as differences in actual social contribution.

## 1. Fields of study

| Major family | Records | Positive-label rate | Difference from STEM |
|---|---:|---:|---:|
| Health | 3,464,814 | 37.14% | +20.59 pp |
| Education | 2,696,674 | 33.74% | +17.19 pp |
| Agriculture / natural resources | 644,651 | 27.17% | +10.61 pp |
| Liberal arts / social sciences / arts | 10,852,507 | 25.90% | +9.34 pp |
| Other fields | 7,124,321 | 24.52% | +7.97 pp |
| Business | 9,254,252 | 17.76% | +1.21 pp |
| STEM | 9,096,411 | 16.55% | baseline |
| Trades / transportation | 555,028 | 13.16% | -3.40 pp |

The liberal-arts-major result is not only a gender-composition effect. Women score above men in every reported major family, but liberal arts remains above STEM for both sexes:

| Major family | Female | Male |
|---|---:|---:|
| Health | 39.75% | 31.13% |
| Education | 38.47% | 26.32% |
| Liberal arts / social sciences / arts | 30.64% | 19.95% |
| Other fields | 29.43% | 19.26% |
| STEM | 23.91% | 13.48% |
| Business | 20.70% | 15.24% |

![Positive-label rate by field of study](analysis_outputs/charts/major_families.png)

## 2. Degree level

| Highest reported degree | Records | Positive-label rate |
|---|---:|---:|
| Doctorate | 2,192,321 | 31.12% |
| Master's | 7,366,908 | 25.68% |
| Certificate | 500,571 | 19.21% |
| Bachelor's | 11,983,712 | 18.75% |
| Associate | 1,619,232 | 17.30% |
| Other / unspecified | 4,802,640 | 16.52% |

Advanced degrees show a steep gradient, but this can reflect occupation, sector, biography length, and selection into graduate education.

## 3. Employer type and size

### Employer type

| Employer type | People | Positive-label rate |
|---|---:|---:|
| Nonprofit | 2,019,156 | 36.90% |
| Educational | 1,032,906 | 33.58% |
| Government | 1,013,834 | 25.38% |
| Private | 12,001,951 | 19.56% |
| Public subsidiary | 1,059,666 | 14.38% |
| Public | 3,998,732 | 13.21% |

Within broad occupation, the weighted gaps are:

| Comparison | Occupations | Positive occupations | Weighted gap | Median gap |
|---|---:|---:|---:|---:|
| Nonprofit minus private | 76 | 64 | +10.33 pp | +5.65 pp |
| Government minus private | 50 | 43 | +4.19 pp | +4.72 pp |
| Educational minus private | 44 | 31 | +3.65 pp | +3.47 pp |
| Private minus public | 95 | 75 | +2.32 pp | +1.54 pp |

This is one of the most robust results: the nonprofit association is not explained solely by nonprofits employing more counselors, teachers, or health workers.

### Employer size

| Employees | People | Positive-label rate |
|---|---:|---:|
| 1–10 | 3,211,693 | 27.22% |
| 11–50 | 2,182,849 | 22.61% |
| 51–200 | 2,324,458 | 20.75% |
| 201–500 | 1,653,661 | 20.99% |
| 501–1,000 | 1,350,024 | 20.84% |
| 1,001–5,000 | 3,047,404 | 20.45% |
| 5,001–10,000 | 1,331,539 | 19.59% |
| 10,001+ | 6,024,617 | 16.84% |

Small employers (1–50) exceed large employers (5,001+) in 107 of 122 comparable occupations. The weighted within-occupation gap is **+3.84 points**.

![Employer findings](analysis_outputs/charts/employers.png)

![Within-occupation robustness](analysis_outputs/charts/within_occupation.png)

## 4. Colleges

The IPEDS/Carnegie crosswalk matched 3,676 of 6,422 Snowflake institutions with at least 500 records by unique exact name (57.2%). Of these, 2,269 are matched four-year institutions. Results below use person-school records, so a person with multiple schools can appear more than once.

### Liberal-arts hypothesis

| Group | Institutions | Person-school records | Positive-label rate |
|---|---:|---:|---:|
| Liberal-arts colleges | 208 | 1,507,752 | 26.19% |
| Other four-year colleges | 2,061 | 32,497,054 | 24.27% |

- Person-weighted gap: **+1.92 percentage points**.
- College-weighted gap: **+0.44 points**.
- Restricting to institutions with at least 2,000 records: **+2.00 points person-weighted** and **+1.44 points college-weighted**.

The liberal-arts advantage appears in every selectivity band:

| Selectivity | Liberal arts | Other four-year | Gap |
|---|---:|---:|---:|
| Elite, under 10% admitted | 29.06% | 25.98% | +3.08 pp |
| Highly selective, 10–25% | 25.55% | 23.81% | +1.74 pp |
| Selective, 25–50% | 27.91% | 23.78% | +4.13 pp |
| Moderately selective, 50–75% | 26.43% | 22.99% | +3.44 pp |
| Less selective / open | 25.17% | 24.70% | +0.47 pp |

![Liberal-arts college differences](analysis_outputs/charts/liberal_arts_selectivity.png)

### Other institutional findings

- **HBCUs:** 82 matched institutions, 568,064 person-school records, 29.39% positive versus 24.27% for 2,187 non-HBCUs; gap **+5.12 points**.
- **Control:** private nonprofit 25.51%, private for-profit 25.82%, public 23.45%. College-weighted results change the ordering: private nonprofit 28.11%, public 24.00%, private for-profit 23.70%.
- **Selectivity is not monotonic:** elite institutions lead at 26.07%, but less-selective/open institutions are next at 24.71%; moderately selective institutions are lowest at 23.17%.
- **Carnegie groups:** mixed special-focus (37.13%) and baccalaureate/associate (36.01%) are highest; doctoral/professional (27.78%) and baccalaureate arts and sciences (26.19%) follow. These very high categories may reflect unusual institution mix and should be inspected school by school.

## 5. Occupations and industries

### Highest occupations, minimum 5,000 people

| Occupation | People | Positive-label rate |
|---|---:|---:|
| Chiropractors | 5,627 | 61.81% |
| Marriage and family therapists | 8,903 | 60.06% |
| Counselors, all other | 6,138 | 58.37% |
| Mental health counselors | 17,255 | 58.02% |
| Substance-abuse and behavioral-disorder counselors | 7,666 | 57.19% |
| Fitness and wellness coordinators | 5,642 | 56.42% |
| Healthcare social workers | 8,549 | 55.92% |
| Mental-health and substance-abuse social workers | 8,739 | 55.12% |
| Dietitians and nutritionists | 19,678 | 54.62% |
| Clinical and counseling psychologists | 40,492 | 54.41% |

### Lowest occupations, minimum 5,000 people

| Occupation | People | Positive-label rate |
|---|---:|---:|
| Computer network architects | 14,927 | 5.59% |
| Database administrators | 10,727 | 6.67% |
| Cost estimators | 10,077 | 6.68% |
| Database architects | 6,823 | 7.09% |
| Computer programmers | 26,015 | 7.23% |
| Electronics engineers, except computer | 6,095 | 7.28% |
| Telecommunications engineering specialists | 6,233 | 7.32% |
| Electricians | 10,028 | 7.93% |
| Computer systems engineers / architects | 116,290 | 8.02% |
| Film and video editors | 6,876 | 8.04% |

### Highest industries

Mental health care (53.77%), philanthropy (49.23%), professional training and coaching (49.06%), civic and social organizations (48.54%), individual and family services (45.00%), nonprofit management (44.42%), public policy (44.25%), primary/secondary education (43.49%), health/wellness/fitness (43.42%), and alternative medicine (42.41%).

### Lowest industries

Semiconductors (7.08%), electrical/electronic manufacturing (8.60%), plastics (8.66%), machinery (8.86%), shipbuilding (8.94%), printing (8.95%), railroad manufacture (9.01%), oil and energy (9.04%), industrial automation (9.08%), and investment banking (9.42%).

### Broad job roles

Health (44.26%), education (38.38%), and public service (37.91%) lead. Engineering (10.12%), trade (10.79%), finance (11.79%), analyst (13.29%), sales engineering (13.61%), and fulfillment (14.55%) are lowest.

These rankings are consistent with the model's construct, but they may also reveal that the classifier recognizes explicit helping-language more readily than indirect contribution through technical or commercial work.

## 6. Demographics

### Gender within occupation

Female profiles exceed male profiles in **94 of 96** broad occupations, with a median gap of **+5.73 points**. The largest gaps include natural-sciences managers (+15.43), counselors (+15.34), physical-sciences postsecondary teachers (+15.30), training and development managers (+14.93), chief executives (+14.05), and actors/producers/directors (+13.94). The only reversals are paralegals/legal assistants (-2.03) and financial analysts/advisors (-1.50); accountants/auditors are essentially tied (+0.06).

### Race within occupation

Among occupations with sufficient observations for both groups:

| Comparison | Occupations | Positive gaps | Median gap | Weighted gap |
|---|---:|---:|---:|---:|
| Black minus White | 46 | 41 | +2.60 pp | +4.71 pp |
| Hispanic minus White | 52 | 45 | +0.60 pp | +0.85 pp |
| Asian minus White | 43 | 5 | -2.97 pp | -3.58 pp |

These fields may be inferred or incomplete. Because the model reads profile text, language and profile-completeness differences can create measurement bias. Treat these as an audit signal, not a claim about inherent group differences.

### Birth cohort

Raw positive-label rates rise from 17.82% for the 1950s cohort to 22.92% for the 1990s cohort. The comparison survives occupation adjustment: 1980–1999 exceeds 1950–1969 in 42 of 43 occupations, with a weighted gap of **+4.74 points** and a median of **+4.35**.

## 7. Geography

### Highest states

Vermont (27.78%), Hawaii (26.39%), Montana (25.49%), Oregon (25.46%), Alaska (25.30%), District of Columbia (24.90%), Idaho (24.82%), Maine (24.58%), New Mexico (24.22%), and Wyoming (23.83%).

### Lowest states

New Jersey (17.67%), Texas (18.91%), Alabama (19.46%), New York (19.64%), Georgia (19.84%), Louisiana (19.92%), Florida (20.09%), Mississippi (20.25%), Illinois (20.25%), and Virginia (20.40%).

### Metro extremes, minimum 5,000 people

Highest: Rexburg, ID (35.30%); Flagstaff, AZ (30.03%); Amherst–Northampton, MA (29.82%); Eugene–Springfield, OR (28.96%); Bloomington, IN (28.89%); Hilo–Kailua, HI (28.77%); and Bellingham, WA (28.76%).

Lowest: Watertown–Fort Drum, NY (13.53%); Odessa, TX (14.26%); Midland, TX (14.33%); Houma–Bayou Cane–Thibodaux, LA (15.27%); San Jose–Sunnyvale–Santa Clara, CA (15.28%); Lake Charles, LA (15.79%); and Huntsville, AL (16.18%).

Geographic results likely reflect local industry and occupation composition. University-centered and helping-sector metros rise; technology, energy, manufacturing, and military-centered metros fall.

## 8. Model behavior and limitations

### Summary-length effect

| Profile-summary length | People | Positive-label rate |
|---|---:|---:|
| Under 100 characters | 1,837,775 | 6.38% |
| 100–249 | 5,063,412 | 10.97% |
| 250–499 | 9,247,041 | 14.46% |
| 500–999 | 8,379,876 | 26.29% |
| 1,000+ | 5,167,735 | 38.57% |

The 1,000+-versus-under-100 gap is **32.19 points**. Longer text creates more opportunities to mention volunteering, public service, health, education, impact, and related language. It also correlates with education, occupation, age, seniority, and platform behavior. Future analysis should adjust or stratify by summary length before presenting institution-level rankings as substantive outcomes.

![Summary-length effect](analysis_outputs/charts/summary_length.png)

### Probability polarization

| Probability band | People | Share |
|---|---:|---:|
| Under 0.001 | 19,336,907 | 65.12% |
| 0.001–0.01 | 2,509,156 | 8.45% |
| 0.01–0.10 | 876,860 | 2.95% |
| 0.10–0.50 | 495,873 | 1.67% |
| 0.50–0.80 | 271,682 | 0.91% |
| 0.80–0.99 | 661,182 | 2.23% |
| 0.99–1.00 | 5,544,179 | 18.67% |

The model behaves more like a high-confidence detector than a smooth ranking measure. The mean and the median therefore tell very different stories, and small changes to the decision rule could affect the labeled share around the threshold even when the extreme predictions remain unchanged.

### What can and cannot be claimed

- These are **associations in model outputs**, not causal effects on social contribution.
- The model measures what is legible in profile text. It may undercount technical, operational, informal, or unpublicized contributions.
- People select into colleges, majors, occupations, sectors, and locations. The within-occupation checks reduce one source of compositional confounding but do not remove it.
- Education rows are not necessarily a person's primary or completed institution, and people can contribute multiple person-school records.
- Exact-name IPEDS matching covers 57.2% of eligible institution names. Unmatched and ambiguously named schools may differ systematically.
- Very large samples make conventional p-values nearly automatic. Effect size, robustness, and measurement validity matter more here.
- Demographic comparisons require a fairness audit because both profile availability and text style may vary by group.

## 9. Recommended next analyses

1. Re-estimate the college, major, employer, gender, race, and cohort gaps within summary-length bands or with summary length as a covariate.
2. Fit a person-level multivariable model with occupation, industry, employer type/size, education, geography, cohort, gender, and text length together.
3. Restrict college analysis to a defined primary undergraduate institution and one row per person.
4. Add a curated review step for high-volume unmatched institutions, while retaining a strict match-quality flag.
5. Independently annotate a stratified audit sample across occupations and summary lengths to estimate precision, recall, and subgroup calibration in the deployment population.
6. Inspect false negatives in engineering, finance, manufacturing, and technical roles to determine whether the label definition or training language misses indirect forms of societal contribution.

## Conclusion

The data supports a careful version of the original hypothesis: **people associated with liberal-arts colleges are modestly more likely to receive the model's societally-good label, and that advantage appears across selectivity bands.** The larger story is that health and education fields, nonprofit work, smaller organizations, advanced degrees, female profiles, and younger cohorts all correlate much more strongly with the label. The nonprofit and cohort patterns survive broad occupation adjustment. However, the model's extreme dependence on profile-summary length means the results should be presented as promising descriptive evidence that requires text-length adjustment and human validation—not as a causal ranking of people or colleges.
