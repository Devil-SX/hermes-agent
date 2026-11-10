# External API authorization

A packaged integration can configure `platforms.api_server.authorization_provider`
as `module:callable` instead of `API_SERVER_KEY`. The two modes are mutually
exclusive. The callable is trusted configuration loaded at adapter construction;
missing modules or invalid callables fail startup without a static-key fallback.

For each request, the profile middleware calls it in a worker thread with keyword
arguments `method`, `path`, `authorization` and `profile`. Return `None` for an
invalid credential, raise `PermissionError` for an unauthorized operation, or
raise another exception when authority verification is unavailable. These become
401, 403 and 503 respectively. Successful results must be dictionaries containing
a stable, nonempty `principal_id` of at most 256 characters. Optional `caller` and
`operations` discovery metadata appear in `/v1/capabilities` under `auth`; never
include credentials. The identity also scopes browser-control principal derivation.

The integration must check current grants and revocation on every request and
authorize both profile and operation. The middleware stores the result only on
that request; existing handler checks consume it without another network lookup.
The configured provider governs the listener's entire route surface, including
handlers which otherwise have public responses. CORS processing does not grant
business access. The stock shared-key mode and its startup checks remain unchanged
for installations that do not configure an external provider.
