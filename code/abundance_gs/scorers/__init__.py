"""Material scorers of the benchmark, copied from the scripts the paper ran (see third_party/THIRD_PARTY.md). Run as
`python -m abundance_gs.scorers.<name>` by stage eval; a new scorer is a module here plus its name in SCORERS."""

SCORERS = ('benchmark', 'song_study', 'song_office', 'protocol', 'merged', 'nvidia', 'city')
