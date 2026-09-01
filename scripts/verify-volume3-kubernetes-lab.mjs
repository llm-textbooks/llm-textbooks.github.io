import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import YAML from "yaml";

const ROOT = path.resolve(import.meta.dirname, "..");
const LAB = path.join(ROOT, "labs/volume-3/kubernetes");
const BASE = path.join(LAB, "base");
const failures = [];
const checks = [];

function docs(file) {
  return YAML.parseAllDocuments(fs.readFileSync(file, "utf8"))
    .map(d => d.toJSON())
    .filter(Boolean);
}
function loadBase() {
  const k = docs(path.join(BASE, "kustomization.yaml"))[0];
  const all = [];
  for (const resource of k.resources) all.push(...docs(path.join(BASE, resource)));
  return all;
}
function one(objects, kind, name) {
  const found = objects.filter(x => x.kind === kind && x.metadata?.name === name);
  if (found.length !== 1) failures.push(`expected one ${kind}/${name}, got ${found.length}`);
  return found[0];
}
function require(condition, message) {
  checks.push({ message, pass: Boolean(condition) });
  if (!condition) failures.push(message);
}
function selectorMatches(deploy) {
  const selected = deploy.spec?.selector?.matchLabels ?? {};
  const labels = deploy.spec?.template?.metadata?.labels ?? {};
  return Object.entries(selected).every(([k, v]) => labels[k] === v);
}

const objects = loadBase();
const namespace = one(objects, "Namespace", "agent-lab");
const serviceAccount = one(objects, "ServiceAccount", "agent-api");
const role = one(objects, "Role", "agent-api-read-contract");
const binding = one(objects, "RoleBinding", "agent-api-read-contract");
const quota = one(objects, "ResourceQuota", "agent-lab-budget");
const service = one(objects, "Service", "agent-api");
const deploy = one(objects, "Deployment", "agent-api");
const pdb = one(objects, "PodDisruptionBudget", "agent-api");
const policies = objects.filter(x => x.kind === "NetworkPolicy");
const container = deploy?.spec?.template?.spec?.containers?.[0];

require(namespace?.metadata?.labels?.["lab.llm-textbooks.io/disposable"] === "true", "namespace must be marked disposable");
require(serviceAccount?.automountServiceAccountToken === false, "ServiceAccount token automount must be disabled");
require(role?.rules?.length === 1 && role.rules[0].resources?.[0] === "configmaps" && role.rules[0].verbs?.join(",") === "get", "RBAC must grant only named ConfigMap get");
require(binding?.subjects?.[0]?.name === "agent-api" && binding?.roleRef?.name === "agent-api-read-contract", "RoleBinding must bind agent-api to narrow Role");
require(quota?.spec?.hard?.pods === "6" && quota?.spec?.hard?.["requests.cpu"] === "500m", "ResourceQuota must bound pod and CPU request budget");
require(policies.length === 2, "base must contain default-deny plus ingress allow NetworkPolicy");
require(policies.some(p => p.metadata?.name === "default-deny-ingress-egress" && p.spec?.policyTypes?.includes("Ingress") && p.spec?.policyTypes?.includes("Egress")), "default-deny must select ingress and egress");
require(policies.some(p => p.metadata?.name === "allow-agent-api-from-lab" && p.spec?.podSelector?.matchLabels?.["app.kubernetes.io/name"] === "agent-api"), "ingress policy must target agent-api labels");
require(service?.spec?.selector?.["app.kubernetes.io/name"] === "agent-api" && service?.spec?.ports?.[0]?.targetPort === "http", "Service must select agent-api and named HTTP port");
require(selectorMatches(deploy), "Deployment selector must match Pod template labels");
require(deploy?.spec?.replicas === 2 && deploy?.spec?.strategy?.rollingUpdate?.maxUnavailable === 0 && deploy?.spec?.strategy?.rollingUpdate?.maxSurge === 1, "Deployment must use two replicas and zero-unavailable rolling update");
require(deploy?.spec?.template?.spec?.serviceAccountName === "agent-api" && deploy?.spec?.template?.spec?.automountServiceAccountToken === false, "Pod must use tokenless agent-api ServiceAccount");
require(deploy?.spec?.template?.spec?.terminationGracePeriodSeconds === 30 && Boolean(container?.lifecycle?.preStop?.exec?.command?.length), "Deployment must declare grace period and preStop hook");
require(container?.readinessProbe?.tcpSocket?.port === "http" && container?.livenessProbe?.tcpSocket?.port === "http", "Deployment must declare named-port readiness and liveness probes");
require(container?.resources?.requests?.cpu && container?.resources?.limits?.memory, "container must declare resource requests and limits");
require(container?.securityContext?.allowPrivilegeEscalation === false && container?.securityContext?.readOnlyRootFilesystem === true, "container must use restricted security context");
require(pdb?.spec?.minAvailable === 1 && pdb?.spec?.selector?.matchLabels?.["app.kubernetes.io/name"] === "agent-api", "PDB must preserve one selected agent-api Pod");

const defect = docs(path.join(LAB, "overlays/intentional-defect/kustomization.yaml"))[0];
const defectPatch = defect?.patches?.[0]?.patch ?? "";
const defectSelector = defectPatch.match(/value: ([^\n]+)/)?.[1]?.trim();
const defectDetected = defectSelector === "agent-api-defective-selector" && defectSelector !== deploy?.spec?.template?.metadata?.labels?.["app.kubernetes.io/name"];
require(defectDetected, "intentional-defect overlay must be rejected: selector does not match template label");

const commands = {};
for (const binary of ["kubectl", "kustomize", "kubeconform", "kind"]) {
  const probe = spawnSync("sh", ["-lc", `command -v ${binary} || true`], { encoding: "utf8" });
  commands[binary] = probe.stdout.trim() || null;
}
const report = {
  schemaVersion: 1,
  lab: "volume3-kubernetes-staging",
  executionKind: "local YAML contract validation; no Kubernetes cluster, API server, CNI, image pull, Pod, receiver, or external tool call",
  clusterExecution: false,
  staticChecks: checks,
  intentionalDefect: {
    overlay: "labs/volume-3/kubernetes/overlays/intentional-defect",
    expected: "Deployment selector/template label mismatch is rejected",
    detected: defectDetected,
  },
  toolsFound: commands,
  failures,
};
fs.mkdirSync(path.join(ROOT, "reports"), { recursive: true });
fs.writeFileSync(path.join(ROOT, "reports/volume3-kubernetes-lab-validation.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ lab: report.lab, checks: checks.length, failures: failures.length, clusterExecution: false, toolsFound: commands }, null, 2));
if (failures.length) process.exit(1);
