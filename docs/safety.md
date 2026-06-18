# Safety

Kratos is allowed to edit and test local projects, so command and path safety
are centralized in `kratos/safety.py`.

Every shell execution path and every file-writing path must call the guard.
This includes normal verification commands, inspect commands, and coder file
operations.

## Permission Profiles

The CLI exposes `/permission low|mid|high`. Permission profiles influence how
ambitious shell work may be, but they do not disable the hard safety blocklist.

Even in a permissive profile, commands blocked by `SafetyGuard` are not run.

## Blocked Command Classes

The command guard is Windows-first but also covers common POSIX patterns.

Blocked categories include:

- disk formatting and raw disk writes, such as `format`, `mkfs`, `dd`, and
  `diskpart`;
- recursive forced deletion patterns, such as `rm -rf`, `rd /s`, `rmdir /s`,
  `del /s /q`, and unsafe `Remove-Item -Recurse/-Force`;
- shutdown, reboot, and power-state changes;
- registry writes and deletes;
- scheduled task creation or modification;
- user and group administration through commands such as `net user`;
- boot and shadow-copy modification commands;
- download-and-execute chains;
- `Invoke-Expression`, `iex`, encoded commands, and Base64 command execution;
- hidden elevation and hidden `Start-Process` usage;
- credential scraping or LSASS access patterns;
- direct access to token, secret, password, or API-key environment values;
- broad ACL or ownership changes such as `takeown` and unsafe `icacls` grants;
- obvious shell bombs and equivalent destructive patterns.

Blocked commands return a structured result with `blocked: true`, a block
reason, and exit code `126`.

## Path Guard

`check_path` enforces project-root confinement for writes and deletes:

- relative escapes such as `../../outside.txt` are blocked;
- absolute paths outside the project are blocked;
- `.git/objects`, `.git/refs`, and `.git/HEAD` are blocked.

The search and read layers also ignore heavy/generated directories and skip
binary or oversized files.

## Web Guard

The web layer uses direct HTTP helpers, not shell download commands.

Rules:

- only `http` and `https` URLs are accepted;
- private, local, and loopback literal hosts are rejected;
- responses are size-capped;
- requests time out;
- content is treated as data;
- successful fetch/search actions are logged as research evidence.

## Defense In Depth

Safety is applied in multiple places:

- `execution/tools.py` checks file paths before writes/deletes.
- `execution/shell.py` checks commands before running them.
- `core/buildtest.py` checks verification commands again before execution.
- `verification.py` keeps an allowlist-style view of expected build/test
  command prefixes.

The result is intentionally conservative. When Kratos cannot prove that a
command is safe enough for the current profile, it should refuse or report a
partial result instead of forcing execution.
