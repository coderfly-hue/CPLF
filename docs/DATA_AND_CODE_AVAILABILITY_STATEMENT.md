# Manuscript Data and Code Availability wording

The following paragraph is synchronized with the revised LaTeX manuscript's
`Data Availability` section. The GitHub URL must be verified after the author
publishes the final repository and fixed release version; this ZIP is a local
release candidate, not proof of public GitHub publication.

> The six benchmark datasets are obtainable from their respective original
> sources, subject to provider terms; third-party raw datasets are not
> redistributed. The CPLF-LLM code and manuscript-associated numerical
> materials are described at https://github.com/coderfly-hue/CPLF. The
> repository includes the executable reference implementation, configuration
> and dependency files, five random seeds (2024--2028), command-level
> training and evaluation instructions, an Algorithm 1 step-to-function map,
> strict dataset preprocessing adapters, reported seed-level and
> paired-comparison files, and scripts to regenerate selected numerical tables
> and empirical figure displays. The README identifies the exact protocol
> subdirectory. Data adapters require author-verified source-field mappings
> and dataset-specific decisions not fixed in the manuscript, while some
> figures use explicitly labelled transcriptions of published table or
> display values. The published case exports lack complete original
> GPU/log/checkpoint provenance, raw evaluator records and certain ID
> mappings. Thus these materials permit source inspection and offline
> numerical/display consistency checks, but do not by themselves certify an
> independent end-to-end reproduction of every reported experiment. The
> repository documents the remaining inputs and evidence required for full
> independent replication.

After uploading, add a commit hash or immutable release URL to the final
manuscript if the journal allows it, and ensure that the PDF is regenerated
from the committed LaTeX source.
