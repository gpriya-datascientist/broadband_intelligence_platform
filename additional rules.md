1a — Write the host/port splitting code
 app stores its address as one combined value ("127.0.0.1:8000"). cli_serve.py needs new code that splits this into a separate host and port before starting the server. Use a proper address-parsing tool, not a simple "split at the colon"  a plain colon-split breaks on certain valid address formats (IPv6) that contain more than one colon.

1b — Explicitly set the server to use exactly one worker
When starting the server, explicitly tell it to run as a single process (workers=1), instead of leaving that unset. This matters because right next to this code, the plan already adds a safety guard for a Windows problem caused by a server accidentally spawning multiple worker processes  explicitly forcing one worker removes that risk entirely instead of just guarding against it after the fact.

1c — Add an automatic re-check for the torchvision exclusion
The build process currently skips including a certain AI-related library because "it's not needed"  verified once, by hand. Add a step to testing so this gets re-confirmed automatically every time the app is built, not just trusted forever from a one-time check. Otherwise, if a future code change ever starts needing that library, the built app would still seem to work  right up until someone actually uses the feature that needs it, at which point it would fail unexpectedly.

2a) What happens if the backend never becomes healthy within the 120-second wait?
The plan says pvl serve polls the backend's health check for up to ~120 seconds — but it never says what happens if that time runs out and the backend still isn't ready. Does it clean up the backend process it already started? Does it print a clear error? Or does it just leave a broken, half-started backend running in the background silently? This needs an explicit answer.

2b) Checking "is it running" by PID alone is a known, real risk
pvl status/pvl stop are described as checking a saved process ID number against whether something's running. The problem: on both Windows and Linux, process ID numbers get reused once a program closes — so if your backend crashed a while ago, the operating system could hand that exact same ID number to a completely unrelated program later. Checking only "does something exist with this ID" isn't enough — it needs an extra check (like also confirming the running program's name matches) to avoid falsely reporting an unrelated program as "your backend is running."


2c) The zip-diagnostics tool (pvl doctor --bundle) might not work reliably on Mono specifically
The plan mentions using a built-in tool to create zip files, describing it as "in-box on net48/Mono" — but the compression support built into Mono has historically had inconsistent behavior across different Mono versions. Since this whole plan explicitly develops and tests on Mono before ever touching real Windows, this specific feature is worth testing directly and early, rather than assuming it behaves identically to real .NET Framework.

3a) The 15-second wait may not be enough. We already know from earlier discussion that the very first startup can take up to ~120 seconds because of database migrations and model loading — a mid-operation shutdown (say, while training/optimizing) could plausibly need longer than 15 seconds to wrap up safely. Worth confirming this number against a real worst-case, not just picking it arbitrarily.

3b) "Log files that auto-rotate by size" needs care on Windows specifically
The plan says log output gets saved into files that automatically rotate once they get too big. This sounds simple, but doing this correctly while a file is actively being written to is trickier on Windows than on Linux — Windows can refuse to let you rename or replace a file that's still open/in-use by an active writer, which Linux normally allows without issue. This detail isn't addressed, and it's specifically a Windows-vs-Linux difference worth testing directly, not assuming will "just work" the same on both

3c) Frontend shutdown is treated as "just kill it," which is probably fine — but worth confirming it truly has no unsaved state (e.g., no pending writes, no open upload streams) before assuming a plain kill is always safe.

