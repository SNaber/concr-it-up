# Single-VM deployment overview

The repository includes example files for running KNN-ConcrItUp on a single
Linux VM. The intended topology is:

- Nginx terminates TLS and applies request and rate limits;
- Gunicorn serves the Flask application on localhost;
- one separate worker consumes the durable SQLite job queue;
- embedding models are mounted read-only;
- job inputs and outputs are stored on persistent writable storage.

The templates are under `deploy/bwcloud/` and assume a Python 3.11 environment.
They are examples rather than a complete infrastructure definition and should
be reviewed against the host institution's security, backup, monitoring, and
data-retention policies.

## Installation outline

1. Provision a supported Linux VM and persistent storage with sufficient space
   for the licensed embedding models and temporary job output.
2. Install the repository at a fixed release tag or commit and create its
   virtual environment from `requirements-deploy.txt`.
3. Mount the embedding directory read-only and keep the job database, uploads,
   configurations, and output directories on writable persistent storage.
4. Copy `concr-it-up.env.example` and
   `embedding-allowlist.example.json` to protected system locations. Generate a
   unique application secret and add only embedding files that may legally be
   served.
5. Adapt and install the example systemd units for the web process and worker.
   Exactly one model worker should be active.
6. Adapt the Nginx templates for the chosen hostname, obtain a TLS certificate,
   and proxy only to the local Gunicorn listener.
7. Begin with authenticated access and exercise job submission, restart,
   resource-limit, backup, and restore behavior before considering anonymous
   access.

The example environment limits numerical-library threads and the example
worker unit constrains memory and CPU use. Hosted requests are additionally
subject to server-side input, queue, runtime, output, and retention limits.
Adjusting these values requires testing on the target VM.

## Retention

`CONCRITUP_RETENTION_HOURS` applies to both kinds of user-managed storage:

- terminal job bundles and their database records become eligible that many
  hours after completion or failure;
- owner-scoped uploads and saved configurations become eligible after that
  many hours without a hosted request from the same authenticated identity or
  anonymous browser session.

The worker checks retention according to
`CONCRITUP_CLEANUP_INTERVAL_SECONDS`. A currently running model subprocess can
delay a check until it finishes. Owner directories are locked across the web
and worker processes, skipped while they have preparing, queued, or running
work, and renamed to a contained trash directory before deletion. Read-only
visits do not allocate owner directories. Job submissions copy their inputs
into the job bundle, so later expiration of the original upload does not alter
a queued or retained result.

On the first cleanup after upgrading an older deployment, an owner directory
without activity metadata receives a new activity marker and one complete
retention period rather than being deleted immediately. Clearing an anonymous
browser cookie makes that browser lose access to its namespace but does not
accelerate server-side deletion.

The configured session root is a deletion boundary. It must be a dedicated,
non-symlinked child of the repository `data/` directory and must not overlap
the job root or SQLite database path. The web and worker processes must use
the same session root and retention period.

For an upgrade that changes retention behavior, first drain the queue and stop
both web and worker services. Make a SQLite-safe database backup and back up
the owner-session directory, deploy one fixed release to both processes, then
start the worker and web services from that same release. Do not run old and
new web/worker code together during the upgrade.

## Security and operations

Do not expose the Flask development server or the worker directly to the
internet. Protect the application secret, embedding allowlist, SQLite database,
uploads, and generated artifacts with operating-system permissions. Limit
administrative network access, apply security updates, and monitor service
health, memory pressure, disk use, TLS expiry, and repeated job failures.

Anonymous sessions are not user accounts and should not be presented as
confidential or permanent storage. Publish the configured retention period,
test cross-session isolation, and maintain recoverable backups of persistent
state. Revalidate the synthetic workflow and relevant test suite for every
deployed release.
