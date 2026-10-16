# Documentation

The main README covers installation, profiling, tests, and data units.
Source-adjacent README files describe individual package modules. See
`scripts/README.md` for the public script inventory and external requirements,
and `ASE_DB_FORMAT.md` for database property conventions.

The documentation generator writes its package inventory and version manifest
into this directory:

```bash
uv run atom-dataset-index generate-docs
```
