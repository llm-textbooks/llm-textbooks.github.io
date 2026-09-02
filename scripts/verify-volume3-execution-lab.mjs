import path from "node:path";
import { spawnSync } from "node:child_process";

// Runs the deterministic in-process execution-control fixtures.
// Each harness re-executes its simulation and checks the committed
// event-ledger SHA-256; no network, containers, or external products.
const ROOT = path.resolve(import.meta.dirname, "..");
const runner = path.join(ROOT, "labs/volume-3/execution-control/run_all.py");

const proc = spawnSync("python3", [runner], { encoding: "utf8" });
if (proc.error) {
  console.error(`execution lab runner failed to start: ${proc.error.message}`);
  process.exit(1);
}
process.stdout.write(proc.stdout);
process.stderr.write(proc.stderr);
if (proc.status !== 0) {
  console.error("volume3 execution-control lab FAIL");
  process.exit(1);
}
