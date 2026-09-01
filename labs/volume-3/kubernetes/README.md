# Volume 3 Kubernetes staging manifest

`base`는 Kubernetes API에 적용 가능한 최소 namespace 범위의 배포 묶음이다. ServiceAccount·한정된 RBAC·ResourceQuota·default-deny NetworkPolicy·Service·Deployment·PDB를 함께 둔다. Deployment에는 `maxUnavailable: 0`, `maxSurge: 1`, TCP readiness/liveness, 30초 종료 유예와 `preStop` hook이 있다.

이 manifest가 보장하는 것은 **API 객체의 선언**뿐이다. `busybox` HTTPD는 disposable HTTP fixture이며 AgentRun, policy validator, durable receiver가 아니다. readiness의 TCP 성공도 HTTP port가 열렸다는 뜻일 뿐 schema/policy/receiver compatibility를 증명하지 않는다. 실제 worker는 preStop에서 intake를 닫고 durable drain marker를 남기며 readiness에서 이전 pending record와 receiver contract를 검사해야 한다.

## 정적 검증

저장소 루트에서 다음을 실행한다.

```bash
npm run verify:kubernetes-lab
```

이 명령은 YAML과 base/intentional-defect의 selector·Service·PDB·NetworkPolicy·RBAC·rollout·probe 계약을 읽어 oracle을 실행한다. `kubectl`, `kustomize`, `kubeconform`, `kind`가 설치되어 있으면 해당 검증을 추가할 수 있지만, 이 저장소의 기본 검증은 cluster를 만들거나 API server에 쓰지 않는다.

## kind/staging에서 수동 실행

클러스터와 CNI의 NetworkPolicy enforcement를 사전에 확인한 disposable kind 또는 staging namespace에서만 실행한다.

```bash
kubectl apply -k labs/volume-3/kubernetes/overlays/kind
kubectl -n agent-lab rollout status deployment/agent-api --timeout=120s
kubectl -n agent-lab get deploy,pods,pdb,networkpolicy,resourcequota
kubectl delete -k labs/volume-3/kubernetes/overlays/kind --wait=true
```

`rollout status` 성공은 AgentRun recovery나 network isolation의 증거가 아니다. CNI별 ingress source, DNS egress, actual receiver authorization, drain 중 new-claim=0, receipt reconciliation은 별도의 실제 트래픽·packet·receiver oracle로 검증해야 한다.

## 의도적 결함 oracle

아래 overlay는 Deployment selector가 Pod template label과 일치하지 않도록 만든 **실패용 입력**이다. apply 하지 않는다. Kubernetes API validation은 selector가 template labels를 선택해야 한다. 이 저장소의 verifier는 해당 불일치를 반드시 검출한다.

```bash
npm run verify:kubernetes-lab
# expected: intentional defect rejected by the local selector oracle
```

```bash
# 금지: 실제 cluster에 defective overlay를 apply하지 않는다.
# kubectl apply -k labs/volume-3/kubernetes/overlays/intentional-defect
```
