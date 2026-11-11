# Changelog

## Unreleased

### Fixed

- Snapshot SQLite databases identified by their file header, including extensionless browser databases and external memory-provider stores. Exclude their WAL, shared-memory and rollback-journal files from full and quick backups; preserve ordinary files with similar names.
