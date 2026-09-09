# Prosocial-framing label definition

## Question

Does the self-description use explicit prosocial framing—language that directly emphasizes positive impact on other people, communities, or public welfare?

- `1`: explicit prosocial framing is present.
- `0`: explicit prosocial framing is absent.

## Counts as positive

- Helping or serving people or communities
- An explicit social mission or values-driven purpose
- A named beneficiary such as underserved youth or an at-risk community
- An impact claim tied to public welfare

## Does not count by itself

- Employment in a hospital, nonprofit, school, environmental agency, or other socially oriented setting
- Producing a socially useful product or performing a socially useful occupation
- Customer, client, or stakeholder service in an ordinary commercial relationship
- Mentoring coworkers or helping an employer grow
- Generic phrases such as “results-driven,” “dedicated,” “passionate,” or “impactful”
- Generic patient-focused language without a specific beneficiary, mission, or advocacy statement
- An employer’s mission quoted as organizational context rather than adopted as the speaker’s own framing

## Edge-case policy

1. Judge only language present in the description; do not infer intent from occupation or industry.
2. A product or job activity is not framing unless the description explicitly connects it to beneficiaries or public welfare.
3. Workplace, coworker, client, and customer relationships are not treated as societal beneficiaries.
4. An isolated cause word is insufficient when it appears as résumé boilerplate.
5. Beneficiary language embedded in a role description can qualify when the beneficiary and social purpose are specific.
6. Apply the same strictness consistently across occupations, sectors, and demographic groups.

## Examples

- “Pediatric nurse dedicated to making children and their families feel safe during a stressful time.” → `1`
- “Working to close the digital divide in underserved rural communities.” → `1`
- “Manufacturing engineer in a heart-valve product group focused on process validation.” → `0`
- “Experienced professional in the environmental-services industry.” → `0`
- “Committed to customer satisfaction and helping clients succeed.” → `0`

The label measures framing in text, not the actual societal value of a person, occupation, employer, or institution.
