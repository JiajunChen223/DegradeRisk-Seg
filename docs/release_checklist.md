# Public Release Checklist

Use this checklist before pushing the repository to GitHub or creating a release.

## Required

- Confirm the author metadata in `CITATION.cff`: Jiajun Chen, College of Earth Sciences, Jilin University.
- Confirm the copyright holder in `LICENSE`: Jiajun Chen.
- Add final GitHub repository URLs to `pyproject.toml` if the package will be published.
- Create the public GitHub repository as `DegradeRisk-Seg`.
- Run `python -m pytest`.
- Confirm that `outputs/`, checkpoints, local Plot-Rice archives, manuscript drafts, and generated caches are not staged.
- Confirm that no local paths remain in dependency files.

## Recommended

- Add the final paper DOI to `CITATION.cff` after publication.
- Attach reproduced result summaries or trained checkpoints as GitHub release assets if redistribution is permitted.
- Add a repository description and topics on GitHub.
- Tag the release that corresponds to the submitted or published manuscript.
