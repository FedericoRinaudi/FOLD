# G1-G4 all-sites cohort decision

## Final experimental design

Performance metrics and Random-Forest TreeSHAP retain the original
cross-run cohort: 1,500 held-out sample-run observations, 15 per site,
covering all 100 monitored sites over five repetitions.

Grouped FOLD attribution preserves the original interpretation of G1-G4:
four consecutive balanced blocks of complete ODoH resolutions.

A trace is therefore eligible for grouped FOLD analysis only if it
contains at least four complete ODoH resolutions.

Across the five recorded TEST splits, 92 of the 100 monitored sites have
at least one eligible trace. Eight sites have no eligible held-out trace
and are excluded only from grouped FOLD G1-G4 statistics. They remain in
model-performance evaluation and Random-Forest TreeSHAP analysis.

The fixed FOLD cohort contains up to 15 distinct eligible TEST traces per
site and never reuses a trace ID across repetitions. Selection is
deterministic and balances observations across repetitions. The resulting
cohort contains 1,240 distinct traces from 92 sites.

Seventy-six sites contain the complete quota of 15 distinct eligible
traces and are used as a prespecified sensitivity subset.

## Aggregation

Grouped-FOLD global estimates use:

sample
-> mean within site/repetition
-> mean over the available repetitions of that site
-> equal-weight mean over sites.

Thus a site is not given greater global weight merely because it has more
eligible traces.

The principal grouped-FOLD result uses all 92 eligible sites. A
sensitivity analysis repeats the global aggregation on the 76 sites that
provide all 15 distinct eligible traces.

## Why adaptive G1-Gk grouping was rejected

An alternative was to retain every trace and let short traces use G1 only,
G1-G2, or G1-G3.

This was rejected because the resulting group statistics would not be
directly comparable.

First, G1 and G2 would exist in more observations than G3 and G4. They
would therefore have more opportunities to become the dominant group.

Second, the Shapley efficiency relation distributes the difference between
the explained prediction and its reference value among the active players.
On a trace containing only two groups, that difference is distributed over
two players rather than four. Absolute attribution magnitudes would
therefore not be directly comparable to those obtained in a four-player
game.

Third, the semantic extent of a group changes with the number of groups.
For example, G1 in a two-group trace represents approximately half of the
trace, whereas G1 in a four-group trace represents approximately one
quarter.

Missing G3/G4 values must therefore not be represented by zero, and
adaptive-group explanations are not pooled with fixed G1-G4 explanations.

## Why four packet-position quartiles were rejected

A second alternative was to redefine G1-G4 as four consecutive quarters of
the packet sequence.

This would make four groups available for almost every trace, including
sites lacking four complete resolutions.

It was rejected because it changes the scientific unit of explanation.
G1-G4 would no longer represent blocks of complete ODoH resolutions, but
arbitrary positional quarters of packets.

That would answer a different question and would invalidate direct
comparison with the existing resolution-based Instagram analysis. It would
also require regenerating and reinterpreting every grouped-FOLD result.

The project therefore preserves the resolution-based intervention and
reports its empirical coverage explicitly.

## Why five repetitions were retained

Reducing from five repetitions to three cannot solve the structural
eligibility problem.

A site with no >=4-resolution TEST trace across five repetitions cannot
gain such a trace by discarding two repetitions. Other sites can only lose
eligible observations and run coverage.

Five repetitions therefore maximize usable grouped-FOLD coverage and also
provide better repeated-run uncertainty estimation.

## Pairing

The same fixed eligible FOLD cohort is used for baseline and every defense
configuration. Defense-vs-baseline attribution changes are therefore paired
on experiment_id, site_id, sample_id and attribution dimension.

## Reported coverage

- Performance: 100 monitored sites.
- RF TreeSHAP: 100 monitored sites.
- Primary grouped FOLD G1-G4: 92 monitored sites, 1,240 distinct traces.
- Full-quota grouped-FOLD sensitivity: 76 sites, each with 15 distinct
  eligible TEST traces.
- Eight sites are excluded only from grouped FOLD because no held-out trace
  contains four complete ODoH resolutions.
