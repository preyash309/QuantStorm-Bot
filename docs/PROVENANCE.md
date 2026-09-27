# Provenance and rights

The root workspace initially had no Git repository. A nested checkout at
`test/quantstorm-ps/` tracked `https://github.com/vishwasmiddha/quantstorm-ps`,
branch `main`, HEAD `5fec814`, and contained modified/untracked local work.
Its remote is an upstream reference, not the user's destination repository.
That checkout and all local changes were preserved in the private backup.

Root and `Improvements` engine/config copies were SHA-256 identical before edits.
The nested competition engine/config had different hashes. The root versions
were selected for research compatibility; the reproduced Phase 1 baseline
supports this choice. Upstream competition utilities restore missing modules
referenced by the root backtester. The copied rulebook is useful context;
the executable root configuration and sanitizer take precedence where they
differ, notably response-width behavior.

No explicit LICENSE file was found in the local source/history inventory.
Competition infrastructure, assets, and research adaptations may have distinct
rights. This reconstruction does not assign ownership or grant a license.
Private repository visibility does not resolve third-party redistribution rights.
Sponsor images and the old promotional README are excluded from publication.
Personal participant names, institutions and roll numbers in Python metadata
were replaced with safe example placeholders. Original evidence is in the
private backup, not erased.

The historical action dataset was generated from deterministic reference bots;
it is not a financial dataset. It is retained locally but excluded for size
and provenance caution. The source collector can recreate trajectories.
