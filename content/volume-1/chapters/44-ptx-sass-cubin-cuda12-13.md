# 44장. PTX·SASS·cubin과 CUDA 12.x/13.x: “CUDA 버전이 안 맞는다”를 해체하는 법

서버 process는 정상적으로 떴다. model weight도 GPU에 올라갔고 health check도 통과했다. 그런데 첫 실제 요청이 특정 attention backend에 도달하자 `no kernel image is available for execution on the device`가 나왔다. 같은 container는 이전 GPU에서 잘 돌았다. 담당자는 “CUDA 12 wheel을 CUDA 13 GPU에 올려서”라고 설명했다. 하지만 GPU에는 CUDA 12나 13이라는 label이 붙어 있지 않다. wheel을 만든 toolkit, wheel이 싣고 온 GPU code image, host driver, device compute capability와 framework dispatcher가 서로 다른 판정을 내린다.

이 장의 목표는 오류 문자열을 외우는 것이 아니다. build option이나 environment가 어떤 version source를 읽고, 어떤 predicate로 target을 고르고, 그 결과 어떤 artifact를 만들며, loader·driver가 어느 image를 선택하고, 실패가 어디서 관측되는지를 한 줄로 잇는다. 입문 독자는 44.1~44.4절에서 PTX·cubin·fat binary와 compatibility fixture를 먼저 읽는다. 배포·소스 독자는 44.5~44.7절에서 vLLM·SGLang·FlashInfer·llama.cpp의 실제 gate와 여섯 장애를 따라간다.

## 44.1 서버는 떴는데 첫 요청에서 kernel image가 없었다

왜 startup이 성공했을까. Python package import와 host `.so` load는 GPU kernel launch가 아니다. shared object의 ELF dependency가 해결되고 extension module initialization이 끝나도 특정 template specialization은 첫 request shape에서야 선택될 수 있다. lazy loader는 published cubin을 처음 필요할 때 읽거나, embedded PTX를 driver에 넘겨 JIT할 수 있다. health check가 해당 backend를 실행하지 않았다면 GPU code compatibility는 아직 검증되지 않았다.

사건의 GPU는 새 compute capability였고 wheel 안에는 이전 세대용 native machine code만 있었다. PTX fallback이 있었는지는 확인되지 않았다. filename에는 `cu13`이 들어 있었지만 이것은 package variant 또는 build toolkit 계열의 표지이지 모든 CUDA 13 지원 GPU의 native image 목록이 아니다. `cu13`이라는 이름에서 새 GPU 지원을 추론한 것이 첫 오류였다.

조사는 다섯 좌표에서 시작한다. visible device의 SM, host driver version, wheel 정확한 filename·digest, extension의 code-object/PTX inventory, 선택 backend와 rejection reason이다. `nvcc --version`은 여섯 번째 좌표일 뿐이다. production host에 nvcc가 없어도 prebuilt cubin은 실행될 수 있고, 반대로 새 nvcc가 설치돼도 wheel 내부 image가 저절로 다시 빌드되지 않는다.

첫 번째 대립 가설은 compatible native cubin이 없다는 것이다. 두 번째는 PTX가 있지만 driver JIT가 그 PTX ISA/toolchain을 이해하지 못한다는 것이다. 세 번째는 framework gate가 새 GPU에서 backend를 거부했는데 다른 extension의 오류를 보고 있다는 것이다. 넷째는 code image보다 앞선 shared-library dependency 또는 symbol mismatch다. 모두 “CUDA mismatch”처럼 보이지만 first divergence가 다르다.

이 장에서는 실제 GPU나 compiler를 실행하지 않는다. 고정 official archive와 source, 보존된 distribution inventory로 판정 절차를 만든다. binary 내부를 추출할 도구가 없었던 표본은 filename이나 member count를 code-object 목록으로 과장하지 않는다. 확인하지 못한 것은 미확정으로 남긴다.

## 44.2 source에서 wheel까지: 여덟 층을 정확히 나눈다

### 44.2.1 CUDA source와 PTX virtual target

CUDA C++ 또는 Triton·CuTe DSL source는 programmer가 작성한 입력이다. compiler front end는 target과 option에 따라 중간·device code를 만든다. PTX는 NVIDIA가 정의한 virtual parallel-thread ISA다. `compute_90` 같은 virtual architecture target은 특정 feature 집합을 표현한다. PTX text 또는 embedded PTX는 driver JIT의 입력이 될 수 있다.

PTX를 “GPU assembly”라고 부르면 절반만 전달된다. 인간이 읽을 수 있는 assembly-like syntax이지만 실제 SM이 실행하는 최종 instruction encoding과 같지 않다. `.version`, `.target`, address space와 instruction feature가 있고, driver JIT 또는 `ptxas`가 real architecture용 machine code로 내린다. PTX version과 virtual target도 서로 다른 좌표다.

### 44.2.2 ptxas, cubin과 SASS

`ptxas`는 PTX를 real architecture target에 맞는 GPU machine code로 조립·최적화한다. 결과 code object를 cubin으로 운반할 수 있다. cubin은 특정 target용 ELF 형식 GPU binary이며 symbol, section과 machine code를 담는다. SASS는 이 machine instruction을 사람이 읽는 mnemonic 표현으로 disassemble한 것이다.

따라서 cubin과 SASS text는 같은 것이 아니다. cubin은 binary artifact이고 SASS는 그 안의 code를 표현한 disassembly다. source의 `int4` load가 어떤 SASS width로 내려갔는지는 cubin을 적절한 도구로 검사해야 한다. filename에 `sm90`이 있다고 SASS inventory 전체가 증명되는 것도 아니다.

### 44.2.3 fatbinary와 host shared object

하나의 application이 여러 GPU를 지원하려면 여러 real target cubin과 virtual target PTX를 함께 담을 수 있다. CUDA compilation trajectory는 device compilation 결과를 fat binary에 넣고 host object에 embed하는 경로를 설명한다. loader/runtime는 현재 device에 사용할 image를 찾는다. [CUDA C++ Programming Guide 12.9.1 — Compilation with NVCC, application compatibility](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#application-compatibility)

fat binary는 container이고 cubin·PTX는 그 안의 image 종류다. `.nv_fatbin` section이 host `.so` 안에 embedded될 수도 있고 sidecar file로 배포될 수도 있다. host `.so`는 CPU loader가 여는 ELF shared object이며 `NEEDED`, RPATH/RUNPATH와 host symbol을 가진다. GPU code inventory와 host dynamic dependency는 별도 감사 대상이다.

### 44.2.4 wheel은 package 운반 상자다

Python wheel은 `.py`, metadata, host `.so`, sidecar cubin, JIT source·manifest와 dependency 선언을 운반하는 ZIP 계열 distribution이다. fatbin과 wheel을 같은 것으로 부를 수 없다. wheel 하나에 GPU code가 embedded된 `.so`가 있을 수도, 별도 cubin 수만 개가 있을 수도, core package에는 code가 없고 extra package가 binary를 제공할 수도 있다.

wheel tag와 filename의 `cu129`, `cu130`은 중요한 package identity지만 내부 code image inventory의 대체물이 아니다. CUDA toolkit minor label, bundled `libcudart.so` major, target SM list와 PTX fallback을 각각 확인한다. package manager가 올바른 wheel을 설치했다는 사실과 driver가 실행 가능한 image를 찾았다는 사실 사이에는 여러 단계가 남는다.

### 44.2.5 driver selection과 JIT

kernel launch가 필요할 때 runtime/driver는 device와 code inventory에 맞는 image를 선택한다. native/compatible cubin이 있으면 machine code를 사용할 수 있다. 적절한 cubin이 없고 PTX가 있으면 driver가 PTX를 target device용으로 JIT할 수 있는지 검토한다. 이때 driver가 embedded PTX ISA와 feature를 이해해야 한다.

“PTX가 있으면 미래 GPU도 언제나 된다”는 문장은 틀리다. PTX target이 필요한 feature를 표현하는지, application이 architecture-specific feature를 요구하는지, driver JIT가 PTX/toolchain을 지원하는지, framework가 그 GPU에서 backend를 허용하는지 모두 맞아야 한다. JIT cache가 persistent하지 않으면 first request마다 compile storm이 날 수도 있다.

**한 줄로 이어 쓰는 build-to-runtime ledger.**

각 backend에 다음 여덟 칸을 만든다.

```text
input option/env
→ version source (nvcc / torch build / driver / device SM)
→ build predicate
→ target flags
→ emitted artifact inventory
→ package placement
→ runtime selection/JIT predicate
→ observed backend/error/cache
```

칸이 비었으면 “지원”이라고 결론내리지 않는다. compiler가 `sm_100f`를 이해한다는 사실은 wheel이 그 image를 싣는다는 증거가 아니다. wheel에 image가 있어도 dispatcher가 device·dtype·shape gate에서 거부할 수 있다. dispatcher가 선택해도 driver가 code image를 load하지 못할 수 있다.

## 44.3 SM80·SM90·새 GPU에서 loader 선택을 손으로 계산한다

교육용 wheel에 `sm_80` cubin, `sm_90` cubin과 `compute_90` PTX가 있다고 하자. 실제 distribution inventory라고 오해하지 않는다. 이 fixture의 목적은 가능한 후보를 순서대로 적는 것이다.

### 44.3.1 A100, SM80

A100에서는 `sm_80` native image가 첫 후보다. loader가 symbol과 kernel에 맞는 image를 찾고 host dependency가 해결되며 runtime gate도 허용하면 PTX JIT 없이 실행할 수 있다. wheel을 CUDA 12.9로 만들었는지 13.0으로 만들었는지보다 실제 cubin과 host runtime dependency, driver compatibility가 직접적인 좌표다.

`sm_90` cubin을 SM80에서 쓸 수 있다고 가정하지 않는다. real target machine code는 임의 역호환이 아니다. `compute_90` PTX도 SM80보다 높은 virtual feature target이면 SM80 fallback이라고 볼 수 없다. A100 지원은 `sm_80` image 존재가 핵심이다.

### 44.3.2 H100, SM90

H100에서는 `sm_90` cubin이 후보다. 같은 wheel이 SM80·SM90 native image를 모두 품었으므로 package 하나가 두 GPU에서 JIT 없이 다른 image를 선택할 수 있다. 이것이 fat binary의 가치다. package filename 하나로 내부 target이 하나라고 단정할 수 없는 이유다.

native `sm_90` symbol이 일부 specialization에만 있고 다른 kernel에는 PTX만 있을 수도 있다. inventory는 extension 전체가 아니라 kernel/code object 단위로 볼 수 있다. startup에서 한 kernel이 성공했다고 모든 shape specialization이 준비됐다는 뜻이 아니다.

### 44.3.3 더 새로운 GPU

새 GPU에 exact native image가 없으면 `compute_90` PTX JIT 가능성을 검토한다. driver가 PTX를 받아 새 device용 machine code를 만들 수 있고 source가 architecture-specific restriction을 갖지 않으면 실행 후보가 된다. 하지만 후보일 뿐 보장 표현이 아니다.

embedded PTX가 너무 새로워 installed driver가 이해하지 못하면 unsupported PTX/toolchain 오류가 날 수 있다. 반대로 PTX target이 너무 낮아 새 kernel feature를 표현하지 못하거나 framework가 새 GPU에서 해당 backend를 거부할 수 있다. architecture-specific `a`/family-specific `f` suffix가 있는 target은 plain target과 compatibility 범위를 임의로 바꾸지 않는다.

### 44.3.4 PTX도 compatible cubin도 없는 경우

새 GPU용 native image가 없고 해당 kernel에 PTX fallback도 없으면 `no kernel image` 가설이 강하다. 하지만 error 문자열만 보고 즉시 확정하지 않는다. 실제 selected extension, symbol과 loaded package를 확인한다. 같은 process에 여러 CUDA extension version이 섞여 잘못된 `.so`를 load했을 수도 있다.

손계산 표에는 device SM, exact/compatible cubin 후보, PTX target, driver JIT 가능성, framework gate를 따로 둔다. 한 칸의 “CUDA compatible yes/no”로 합치지 않는다.

**symbol마다 coverage가 다르면 부분 성공이 정상이다.**

하나의 extension 안에 kernel A, B, C가 있다고 하자. A는 `sm_80`, `sm_90`과 PTX를 모두 갖고, B는 `sm_90` native만, C는 runtime JIT source만 갖는다. A100에서 A는 native 후보, B는 unsupported, C는 local compiler가 있을 때만 후보가 된다. H100에서는 A·B native, C JIT다. 새 GPU에서는 A의 PTX JIT 가능성, B no-image, C의 current compiler target 지원을 각각 판정한다.

이 matrix는 서버가 “반쯤” 동작하는 현상을 설명한다. model load의 conversion kernel A는 성공하지만 긴-context attention B가 실패할 수 있다. 일반 GEMM은 library가 처리하지만 새로운 quant C는 JIT compiler 부재로 실패할 수 있다. extension import를 package coverage의 단일 boolean으로 쓰지 않는다.

request shape가 symbol을 결정하는 경로도 넣는다. head dimension, dtype, page size, split 수와 GPU feature가 template key를 바꾼다. production histogram에서 실제로 나오는 key 집합이 coverage audit의 분모다. test에서 128 dimension 하나만 launch해 성공해도 64/96/256 또는 tail specialization은 미검증이다.

coverage를 줄여 안전한 fallback을 선택할 수도 있다. B가 없는 A100에서는 dispatcher가 A 계열 generic kernel로 내려가고 SLO를 충족한다면 supported service path다. 하지만 fallback을 알리지 않고 benchmark에서 B 성능을 기대하면 운영 퇴행이다. manifest에는 preferred, fallback, reject 세 결과를 shape별로 둔다.

PTX coverage도 symbol별이다. fatbinary 안에 PTX image가 있다는 사실과 모든 code가 PTX representation을 갖는다는 사실은 다르다. architecture-specific source가 PTX fallback 없이 real target에만 compile될 수 있다. binary tool이 image target만 보여 주고 symbol mapping을 못 보여 주면 coverage level을 file-level로 표시한다.

JIT source coverage는 source file 존재보다 좁다. runtime generator가 device capability, compiler version, feature option에서 해당 template를 만들 수 있어야 한다. current compiler가 target suffix를 이해하지 못하면 source가 있어도 후보가 아니다. include/header와 dependency가 core wheel에서 빠져도 JIT가 실패한다.

부분 지원 record는 장애 대응뿐 아니라 package size 결정에도 쓴다. 모든 SM·shape cubin을 싣는 wheel은 커진다. 자주 쓰는 key는 published cubin, 드문 key는 PTX/JIT, unsupported old GPU는 fallback처럼 정책을 설계할 수 있다. 중요한 것은 그 경계가 manifest·dispatcher와 배포 compiler 조건에서 일치하는가다.

release gate는 production key coverage를 세 수준으로 보고한다. native AOT, validated JIT/PTX, validated fallback이다. unknown과 hard reject도 따로 센다. “GPU generation 지원”이라는 한 줄 대신 이 분포와 first-request 비용을 제공하면 운영자가 새 GPU rollout 위험을 판단할 수 있다.

## 44.4 CUDA 12.x·13.x compatibility를 네 관계로 나눈다

### 44.4.1 backward compatibility

일반적으로 더 새 NVIDIA driver가 이전 toolkit으로 빌드된 application을 지원하는 경로를 backward compatibility라고 설명한다. 이것은 새 driver가 옛 application을 실행하는 방향이다. application의 bundled libraries와 GPU code가 실제 device를 지원하는지까지 자동 해결한다는 뜻은 아니다.

### 44.4.2 minor-version compatibility

CUDA Compatibility 공식 문서는 major family 안에서 일정 driver 범위로 minor toolkit application을 실행하는 계약을 제시한다. 보존 artifact SHA-256 `7ed42e09da9bd7641257f4b2a07ab44dc0dbbcc41089b08e050cdbb209efe032` 기준 CUDA 12.x minor compatibility 범위는 driver branch 525 이상 580 미만이고, CUDA 13.x는 최소 580이다. 더 새 driver에서는 backward compatibility 관점이 적용될 수 있다. [NVIDIA CUDA Compatibility 13.0.2 — Minor Version Compatibility](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compatibility/index.html#minor-version-compatibility)

이 표는 “CUDA 12 wheel이면 driver 525에서 모든 기능이 된다”는 뜻이 아니다. toolkit과 driver 양쪽에 걸친 신기능은 older driver에서 제한될 수 있다. PTX JIT가 필요한 application은 older driver가 더 새 PTX를 이해하지 못해 실패할 수 있다. minor compatibility를 사용할 때 실제 target architecture code를 포함하라는 지침이 중요한 이유다.

### 44.4.3 forward compatibility package

forward compatibility는 `cuda-compat` user-mode driver library package를 통해 더 새 toolkit application을 특정 data center driver 환경에서 지원하는 별도 배포 방식이다. host kernel driver를 단순히 속이는 환경 변수나 system nvcc 교체가 아니다. 적용 GPU·OS·driver 조건과 package library load path를 공식 matrix로 확인해야 한다. [NVIDIA CUDA Compatibility 13.0.2 — Forward Compatibility](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compatibility/index.html#forward-compatibility)

### 44.4.4 application binary compatibility와 PTX JIT

native cubin compatibility와 PTX JIT는 분리한다. cubin은 real architecture machine code이고 PTX는 driver compiler 입력이다. old driver에서 native image는 load되지만 PTX-only specialization은 실패할 수 있다. 같은 application 안에서도 일부 kernel은 AOT cubin으로 성공하고 늦게 필요한 JIT kernel만 실패하는 이유다.

compatibility record에는 build toolkit/nvcc, bundled cudart major, host driver, compat package, GPU SM, native code target과 PTX target을 각각 적는다. `nvidia-smi CUDA Version` 표시는 driver가 지원하는 최대 API/toolkit 계열의 한 표지이지 현재 Python wheel의 build toolkit이나 installed nvcc를 뜻하지 않는다.

### 44.4.5 compatibility fixture를 행 단위로 판정한다

같은 교육용 wheel을 네 환경에 놓아 보자. wheel의 host `.so`는 `libcudart.so.12`를 필요로 하고, GPU payload에는 `sm_80`, `sm_90` cubin과 `compute_90` PTX가 있다. application이 쓰는 일반 kernel은 세 image에 모두 있고 architecture-specific kernel K는 `sm_90` cubin에만 있다고 가정한다. 이 가정은 symbol별 inventory가 중요하다는 점을 보여 준다.

환경 A는 A100 SM80, CUDA 12.x minor-compatible driver와 올바른 cudart 12다. 일반 kernel은 `sm_80` image 후보가 있다. K는 `sm_90`만 있으므로 A100에서 사용할 수 없다. framework가 K를 A100에서 선택하지 않는다면 service는 정상일 수 있다. 이 결과를 “wheel 전체가 A100 compatible”이라고 쓰기보다 “A100 path의 selected symbols에 sm80 image가 있다”고 쓴다.

환경 B는 H100 SM90와 더 새 driver, cudart 12다. 일반 kernel과 K 모두 `sm_90` native 후보가 있다. backward compatibility 방향으로 새 driver가 옛 runtime application을 지원할 수 있지만 host library가 container에 없거나 wrong RPATH이면 import에서 먼저 실패한다. GPU image가 완전해도 host dependency가 앞선 gate다.

환경 C는 새로운 GPU와 CUDA 12.x old driver다. native image가 없으므로 일반 kernel은 `compute_90` PTX JIT 후보를 본다. driver가 embedded PTX version을 이해하지 못하면 minor compatibility 제한에 걸릴 수 있다. K에는 PTX가 없으므로 일반 kernel이 JIT 성공해도 K는 no-image다. application 일부가 동작하는 현상은 모순이 아니다.

환경 D는 같은 새 GPU에 적절한 새 driver와 cudart 12다. 일반 PTX JIT 가능성은 커지지만 framework device gate가 새 SM을 미지원으로 분류할 수 있다. K는 여전히 image가 없다. driver upgrade는 input code inventory를 만들지 않는다. generic fallback이 있으면 service는 뜨되 성능이 달라질 수 있다.

이 네 행에 CUDA 13 toolkit을 host에 설치해도 prebuilt wheel의 `NEEDED`와 embedded image는 바뀌지 않는다. source JIT path가 호출될 때만 current compiler가 새 artifact를 만들 수 있다. 따라서 toolkit 설치 전후 비교에서 어느 kernel이 AOT이고 어느 kernel이 JIT인지 먼저 분류한다.

이제 wheel 자체를 cu130 variant로 바꿨다고 하자. host `.so`가 `libcudart.so.13`을 필요로 하고 embedded target도 새로 build됐을 수 있다. 그러나 “있을 수 있다”에서 멈춘다. wheel 내부 inventory와 build manifest를 확인해야 한다. package mapping이 cu130이라는 사실만으로 `sm_100f`, `sm_120a` 또는 특정 PTX가 들어 있다고 채우지 않는다.

forward-compat package를 환경 C에 넣는 경우도 별도 행이다. compat user-mode libraries가 공식 지원 matrix에서 host kernel driver와 조합될 수 있는지, application이 요구하는 feature를 지원하는지 본다. `/usr/local/cuda` symlink를 compat path로 바꾸는 것과 같은 임의 조치는 계약이 아니다. loader가 실제 어느 library를 선택했는지 기록한다.

matrix의 결과 칸에는 `import`, `generic native`, `generic PTX JIT`, `special K`, `framework dispatch`를 따로 둔다. 하나라도 unknown이면 overall yes/no를 만들지 않는다. production readiness는 production request가 실제 사용하는 symbol set에서 모든 행이 닫혔는지로 판정한다.

**`sm_100a`, `sm_100f`, `compute_100`을 문자열로 정렬하지 않는다.**

architecture suffix는 version 숫자 뒤 장식이 아니다. architecture-specific target, family-specific target과 virtual target의 feature·compatibility 의미를 NVCC/Programming Guide 해당 판에서 확인한다. plain `sm_100`, `sm_100a`, `sm_100f`를 “더 큰 문자열이 더 호환된다”는 식으로 ordering하지 않는다.

build system source가 CUDA 12.9부터 `100f`를 고르고 이전에는 `100a`를 고른다면 compiler syntax support와 intended family coverage라는 두 이유가 있을 수 있다. 실제 wheel target을 확정하려면 emitted compile flags와 binary inventory가 필요하다. runtime loader가 family image를 어느 device에 사용할 수 있는지는 공식 application compatibility 계약에 묶는다.

PTX virtual target도 feature floor를 가진다. `compute_90` PTX가 새 device에서 JIT될 가능성과 SM90-specific native `sm_90a` code의 compatibility는 같은 규칙이 아니다. architecture-specific feature를 source가 사용하면 generic PTX fallback을 만들 수 있는지도 compile 단계에서 달라질 수 있다.

독자 기록에는 source feature requirement, nvcc target flag, emitted real/virtual image와 runtime device를 네 열로 둔다. target suffix를 생략한 “Blackwell support” 한 칸은 B100/B200/GB 계열과 family target 차이를 숨길 수 있다.

## 44.5 vLLM과 SGLang: build 가능성과 runtime 선택 사이

### 44.5.1 vLLM의 nvcc version source

vLLM build의 첫 version source는 `CUDA_HOME/bin/nvcc -V` 출력이다. `get_nvcc_cuda_version()`은 output에서 `release` token을 찾고 그 뒤 version을 파싱한다. 이 값은 host driver나 `torch.version.cuda`가 아니다. source extension을 현재 build 환경의 compiler로 만들 때 사용하는 toolkit version이다. [vLLM v0.27.1 — `setup.py:1008-1015` — `get_nvcc_cuda_version`](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/setup.py#L1008-L1015)

첫 연쇄를 완성하면 이렇다.

```text
CUDA_HOME
→ CUDA_HOME/bin/nvcc -V release token
→ Version 비교
→ CMake extension 포함·제외와 target flags
→ built .so/fatbinary
→ package wheel
→ import/backend availability
→ selected backend 또는 unavailable reason
```

`CUDA_HOME`을 바꿨지만 이미 받은 prebuilt wheel을 재설치하지 않았다면 embedded artifact는 바뀌지 않는다. 이 environment는 source build 경로의 입력이지 설치된 `.so`를 변환하는 runtime option이 아니다. 반대로 source build에서는 PyTorch의 `CUDA_HOME` resolution과 실제 path를 확인해야 다른 system nvcc를 잘못 읽지 않는다.

vLLM setup은 hosted wheel variant를 CUDA major로 정규화한다. CUDA major 12는 `cu129`, major 13은 `cu130`에 대응시키는 source가 있다. 이것은 호스팅 distribution 이름을 고르는 policy다. CUDA 12.8 local compiler를 썼다고 무조건 `cu128` wheel이 존재한다고 가정하지 않는다. [vLLM v0.27.1 — `setup.py:523-556` — hosted CUDA wheel version mapping](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/setup.py#L523-L556)

두 번째 연쇄는 package selection이다.

```text
requested/observed CUDA major
→ supported mapping {12:cu129, 13:cu130}
→ distribution URL/name
→ downloaded wheel member set
→ host .so dependencies와 embedded/sidecar GPU images
→ device loader/dispatcher
→ 실제 backend
```

mapping까지만 읽고 wheel 내부 `sm_*` target을 추정하지 않는다. package identity가 정해진 뒤 binary inventory를 별도로 감사한다.

### 44.5.2 vLLM extension gate: FA3와 FlashMLA

setup source는 FA3 build를 CUDA 12.3 이상으로 gate하고 FlashMLA를 CUDA 12.9 이상으로 gate한다. predicate가 참이면 extension build 목록에 들어갈 수 있다는 뜻이다. build dependency, architecture target과 compile 성공이 더 필요하다. predicate가 거짓이면 해당 source extension이 wheel에 없을 가능성이 강하지만 runtime fallback이 무엇인지는 별도 dispatcher가 정한다. [vLLM v0.27.1 — `setup.py:1125-1165` — optional CUDA extension gates](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/setup.py#L1125-L1165)

FA3 연쇄는 다음과 같다.

```text
nvcc release from CUDA_HOME
→ >=12.3 predicate
→ FA3 extension build inclusion
→ FA3 symbols/code images in wheel candidate
→ Python import FA3_AVAILABLE
→ device compute capability 9.x predicate
→ FA3 dispatch or explicit unavailable reason
```

runtime interface는 import 성공과 device capability를 또 검사한다. extension이 build됐어도 device가 9.x가 아니면 FA3를 거부한다. [vLLM v0.27.1 — `vllm/vllm_flash_attn/flash_attn_interface.py:20-70` — import와 device gate](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/vllm_flash_attn/flash_attn_interface.py#L20-L70)

이 연쇄에서 `torch.version.cuda`가 들어오는 다른 dependency gate도 있다. PyTorch wheel이 어느 CUDA major로 build됐는지를 나타내는 값과 current nvcc, driver를 구분한다. torch cu12 build 위에 CUDA 13 nvcc를 억지로 결합하면 host headers/library와 extension ABI·dependency가 예상과 달라질 수 있다. 하나의 `CUDA_VERSION` log로 세 값을 덮지 않는다.

### 44.5.3 SGLang AOT policy의 단계

SGLang AOT CMake는 `CUDA_VERSION`을 비교해 default feature와 gencode를 정한다. CUDA 12.4 이상이며 aarch64가 아니면 FA3 기본 enable 후보가 된다. option `SGL_KERNEL_ENABLE_FA3`가 최종 build inclusion을 제어한다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- CUDA 12.8 이상 또는 force option에서는 NVFP4 define을 넣는다.
- [SGLang v0.5.18 — `python/sglang/kernels/aot/CMakeLists.txt:108-177` — FA3 default와 option](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/aot/CMakeLists.txt#L108-L177) [SGLang v0.5.18 — `python/sglang/kernels/aot/CMakeLists.txt:242-252` — FA3/NVFP4 inclusion](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/aot/CMakeLists.txt#L242-L252)

세 번째 연쇄는 NVFP4다.

```text
CMake CUDA_VERSION 또는 SGL_KERNEL_ENABLE_FP4
→ >=12.8 OR force predicate
→ ENABLE_NVFP4 compile definition
→ NVFP4 source/target compiled into extension candidate
→ extension import and device/dtype/model predicate
→ NVFP4 kernel selection 또는 fallback
→ selected backend·latency·numeric observation
```

force option은 compiler·device가 실제 instruction과 target을 지원하지 않아도 성공을 보장하지 않는다. predicate를 우회할 뿐 downstream compile과 runtime gate가 남는다. option 도움말의 “enable”을 support proof로 쓰지 않는다.

### 44.5.4 SM100 family target 표기의 version gate

SGLang CMake는 CUDA 12.9 이상에서 `compute_100f, sm_100f`를 사용하고, 더 이른 지원 경로에서는 `sm_100a`를 둔다. source 주석은 family-specific `f` target이 CUDA 12.9+에서 가능하고 SM100 family를 포괄하는 의도를 밝힌다. CUDA 12.8 이상에는 `compute_120a, sm_120a`도 추가한다. [SGLang v0.5.18 — `python/sglang/kernels/aot/CMakeLists.txt:208-223` — SM100/SM120 gencode](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/aot/CMakeLists.txt#L208-L223)

여기서 `a`, `f`, plain target을 문자열 장식처럼 치환하지 않는다. compiler version이 target syntax를 이해하는지, architecture-specific/family-specific compatibility가 공식 계약상 어디까지인지 확인한다. cu129 wheel이 CUDA 12.8 nvcc를 찾아 JIT하면 `sm_120f` target을 거부할 수 있어 JIT arch resolver가 실제 nvcc를 기준으로 fallback을 정하는 source도 있다. package label과 current compiler가 다를 수 있다는 실제 사례다.

AOT wheel에 target을 실은 것과 runtime CuTe/JIT가 target을 고르는 것은 별도다. SGLang의 FA4 2CTA workaround는 environment option과 `torch.version.cuda` major를 읽어 CUDA 12 codegen regression 경로를 피한다. 이것은 AOT CMake의 nvcc source와 다른 version source다. 두 predicate를 한 표에 쓸 때 `version_source` 열을 반드시 둔다.

### 44.5.5 fallback은 성공일 수도 장애일 수도 있다

FA3가 build되지 않거나 device gate에서 거부되면 다른 attention backend가 요청을 처리할 수 있다. 서버가 정상 응답하므로 compatibility가 완전히 해결됐다고 말할 수 없다. latency·memory·numeric behavior가 달라질 수 있고 CUDA Graph 지원도 다를 수 있다. selected backend와 rejection reason을 관측해야 silent fallback을 발견한다.

반대로 fallback이 정확히 의도된 policy이면 `no kernel image`보다 안전하다. build matrix는 unsupported target을 억지로 싣는 대신 runtime dispatcher가 지원 backend로 내려가게 설계할 수 있다. 운영 판정은 “원하는 backend가 떴는가”와 “서비스가 correctness/SLO를 만족했는가”를 둘 다 기록한다.

### 44.5.6 SGLang JIT resolver가 package와 nvcc를 다시 잇는 이유

SGLang JIT architecture resolver의 주석은 구체적인 혼합 환경을 다룬다. cu129 wheel을 설치했어도 current toolkit이 CUDA 12.8이면 `sm_120f`를 선택할 경우 nvcc 12.8이 target을 거부한다.

그래서 wheel label이 아니라 tvm-ffi와 같은 방식으로 실제 nvcc를 찾고 version에 따라 family-specific `f` 또는 architecture-specific `a` target을 고른다. [SGLang v0.5.18 — `python/sglang/kernels/jit/utils/arch.py:35-88` — current nvcc 기반 architecture resolution](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/jit/utils/arch.py#L35-L88)

이 연쇄는 AOT와 JIT가 같은 package 안에서 다른 target policy를 가질 수 있음을 보여 준다.

```text
installed wheel variant cu129
→ JIT compiler path resolution
→ actual nvcc version 12.8 or 12.9+
→ sm_120a or sm_120f target predicate
→ JIT compile command
→ cached object target
→ current SM module load
→ success/reject and cache key
```

wheel variant만 cache key에 넣으면 CUDA 12.8과 12.9 compiler가 만든 object를 충돌시킬 수 있다. actual compiler version과 selected target suffix를 key에 포함해야 한다. 반대로 compiler path가 같아도 source/template·feature option이 다르면 다른 key다.

FA 2CTA workaround는 또 다른 version source를 보여 준다. environment `FA_DISABLE_2CTA`와 codegen regression 설명이 있고 CUDA major에 따라 path를 제한한다.

이 runtime/JIT source가 `torch.version.cuda`를 읽는다면 current system nvcc를 업그레이드해도 torch build major는 바뀌지 않는다. option이 downstream kernel generation predicate를 어떻게 바꾸는지 확인해야 한다. [SGLang v0.5.18 — `python/sglang/kernels/ops/attention/flash_attn/cute/utils.py:55-80` — 2CTA disable과 CUDA-major gate](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/ops/attention/flash_attn/cute/utils.py#L55-L80)

이 option을 켜면 unsupported toolchain error를 피할 수 있어도 2CTA algorithm·resource behavior가 달라진다. 성공 관측은 compile error 소멸뿐 아니라 selected specialization, output과 performance다. CUDA 13에서 workaround가 불필요해졌다는 주장도 source predicate와 artifact A/B 없이 단정하지 않는다.

JIT compiler를 자동 탐색하는 편의는 재현성 위험을 만든다. container PATH 순서에 `/usr/local/cuda-12.9`, framework-bundled ptxas와 system binary가 함께 있으면 node마다 다른 compiler가 선택될 수 있다. resolver가 longest root 또는 명시적 path를 고르는 규칙을 읽고 resolved absolute path와 digest를 manifest에 남긴다.

AOT artifact가 있어 JIT가 필요 없을 것으로 예상했는데 compiler 탐색 log가 나온다면 missing specialization 또는 cache miss의 신호다. preferred AOT key와 requested runtime key를 비교한다. current compiler를 설치해 증상을 우회하기 전에 왜 published artifact가 hit하지 않았는지 찾는다.

## 44.6 FlashInfer의 core·cubin·JIT 계층과 llama.cpp target

### 44.6.1 core wheel은 kernel inventory 전체가 아닐 수 있다

FlashInfer는 core package와 CUDA-family extra/cubin distribution을 구분한다. 보존된 표본에서 core wheel 4,977 member에는 `.so`, `.cubin`, `.ptx`, `.fatbin` 확장자가 0개였고, cubin wheel 표본에는 23,008 `.cubin` member가 있었다. 이 숫자는 해당 distribution 표본의 member inventory이지 모든 release와 internal code object의 보편 법칙이 아니다.

core import 성공은 필요한 GPU kernel binary가 모두 local이라는 뜻이 아니다. manifest와 package extra, download/cache/JIT 경로가 이어질 수 있다. 반대로 filename에 cubin이 있어도 device·dtype·shape에 맞는 exact symbol이 존재하는지는 manifest lookup까지 확인한다.

**core 다음 여정: manifest lookup에서 load까지.**

cubin loader의 수명은 kernel key/name을 manifest에 조회하고, local packaged artifact 또는 download cache에서 file과 SHA를 확인하고, binary를 load하는 순서다. published cubin miss이면 source/JIT fallback이 가능한 kernel family인지 분기한다. network success와 binary load success, kernel symbol resolution은 다른 완료 지점이다.

네 번째 연쇄는 FlashInfer cubin이다.

```text
op/shape/device key
→ manifest name·expected SHA lookup
→ packaged/local-cache/download candidate
→ integrity check와 cubin load
→ symbol/module availability
→ dispatcher selection
→ cache hit/miss·download·load error observation
```

SHA mismatch를 `no kernel image`와 같은 것으로 처리하지 않는다. integrity 실패는 artifact identity 문제이고, image absence는 target inventory 문제다. download가 막혀도 local JIT가 가능할 수 있지만 compiler와 source package가 필요하다.

### 44.6.2 JIT fallback과 cache key

published cubin이 없으면 JIT path는 architecture list, toolkit/compiler, template parameters와 source digest를 cache key에 넣어야 한다. `FLASHINFER_CUDA_ARCH_LIST` 같은 build input은 nvcc gencode flags를 만든다. multi-arch nvcc 결과는 fatbin 성격을 가질 수 있고 single-arch CuTe DSL object는 target 하나에 묶일 수 있다. 둘의 cache key를 같게 두면 다른 device artifact를 재사용하는 오답이 된다.

Blackwell과 CUDA 13 조합에서 system `ptxas` path를 Triton에 주입하는 predicate는 compiler tool selection의 사례다. package의 CUDA extra, `torch.version.cuda`, current system toolkit과 target device를 구분해야 한다. system ptxas를 바꿨다고 published cubin이 바뀌지는 않으며 JIT 경로에만 영향을 줄 수 있다.

보존된 cu129/cu130 JIT cache 표본에서 `.nv_fatbin`과 각각 `libcudart.so.12`, `libcudart.so.13` dependency가 관찰됐다면 host runtime family 차이의 증거로 쓸 수 있다. 그러나 `cuobjdump`/`nvdisasm`가 없어 internal target inventory를 확인하지 못했다면 filename token으로 `sm_*` 목록을 만들지 않는다. binary audit의 미확정 칸을 그대로 남긴다.

**같은 JIT 여정이 storm으로 갈라지는 조건.**

JIT cache key에 process-unique temp path나 unstable source serialization이 들어가면 같은 kernel도 매 restart마다 miss할 수 있다. cache directory가 ephemeral container layer에 있거나 여러 pod가 공유하지 못해도 first request compile이 반복된다. file은 있지만 compiler/toolkit key가 달라 invalidation될 수도 있다.

JIT storm 조사에는 lookup key, local hit/miss, compile 시작/끝, artifact digest, cache publish와 다음 process hit를 시간순으로 둔다. GPU utilization이 낮고 CPU compiler process가 높다는 현상만으로 확정하지 않는다. download retry, file lock contention 또는 integrity revalidation도 first request를 늦출 수 있다.

### 44.6.3 llama.cpp CMake target과 runtime offload

llama.cpp는 CMake CUDA architecture 설정과 compile feature define으로 어떤 device code를 만들지 정한다. `CMAKE_CUDA_ARCHITECTURES` 또는 project option이 nvcc target에 반영되고, architecture gate에 따라 feature source가 포함된다. NCCL compile option도 distributed code inclusion을 제어한다.

다섯 번째 연쇄는 다음과 같다.

```text
CMake CUDA option/architectures
→ CMake compiler identification과 CUDA version
→ architecture/feature predicates
→ nvcc target flags와 source definitions
→ llama CUDA shared/static artifact
→ runtime backend/device enumeration
→ layer/KQV offload decision
→ selected buffers·kernel launch 또는 CPU fallback
```

컴파일 성공은 model layer가 GPU로 offload됐다는 뜻이 아니다. runtime device availability, user offload option, buffer placement와 graph scheduler가 실제 실행 위치를 고른다. 반대로 CPU fallback으로 서버가 떠도 원하는 CUDA kernel coverage가 검증된 것은 아니다. build target과 runtime offload state를 별도 record로 둔다.

### 44.6.4 FlashInfer source에서 package label과 current compiler를 분리한다

FlashInfer compilation context는 `FLASHINFER_CUDA_ARCH_LIST`가 있으면 공백으로 나눈 architecture 값을 읽는다. AOT entry는 이 environment가 없으면 명시적으로 실패시킨다.

이것은 “현재 build에서 어느 target을 요청했는가”를 재현 가능하게 만드는 계약이다. [FlashInfer v0.6.17 — `flashinfer/compilation_context.py:80-110` — target architecture environment](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/compilation_context.py#L80-L110) [FlashInfer v0.6.17 — `flashinfer/aot.py:825-875` — explicit architecture requirement](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/aot.py#L825-L875)

environment 문자열이 `9.0 10.0`이라면 target set의 입력은 알 수 있지만 실제 emitted fatbin inventory는 compile command와 binary로 확인해야 한다. source generator가 일부 op를 architecture gate에서 제외할 수 있고 compile 실패가 packaging 단계에서 처리될 수 있다. build log와 wheel member를 잇지 않으면 environment 설정만 남는다.

Triton initialization은 system `ptxas`를 찾고 version output을 파싱해 Blackwell·CUDA 13 경로의 environment를 설정한다. 이 predicate는 process runtime에서 JIT compiler path를 선택하는 입력이다. wheel이 어느 CUDA extra로 설치됐는지와 system `ptxas`가 무엇인지 다를 수 있다. [FlashInfer v0.6.17 — `flashinfer/triton/__init__.py:1-25` — Blackwell system ptxas patch](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/triton/__init__.py#L1-L25)

이 연쇄는 여섯 번째로 쓸 수 있다.

```text
system PATH의 ptxas
→ ptxas --version parser
→ CUDA>=13/Blackwell predicate
→ TRITON_PTXAS_BLACKWELL_PATH
→ Triton JIT compile tool
→ generated object/cache key
→ module load
→ compiler error 또는 selected kernel
```

system ptxas가 없으면 published cubin path까지 실패한다고 결론내리지 않는다. 이 연쇄는 Triton JIT에 관한 것이다. cubin manifest hit는 compiler 없이 load될 수 있다. 반대로 ptxas가 있어도 manifest key, source와 target feature가 맞아야 JIT가 성공한다.

cubin download 경로는 download 완료 뒤 각 file checksum을 검증하고 mismatch면 실패한다. integrity check가 있으므로 network에서 byte를 받았다는 사실만으로 cache publish를 완료하지 않는다. [FlashInfer v0.6.17 — `flashinfer/artifacts.py:330-376` — cubin download와 checksum](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/artifacts.py#L330-L376)

### 44.6.5 llama.cpp의 real·virtual target 목록을 읽는다

llama.cpp CUDA CMake는 사용자가 `CMAKE_CUDA_ARCHITECTURES`를 지정하지 않았을 때 compiler version과 build mode에 따라 target 목록을 구성한다. 일부는 `-virtual`, 일부는 `-real` suffix를 사용한다. virtual target은 PTX fallback 성격을, real target은 native code 생성을 요청하는 CMake 표현이다.

목록 하나 안에 둘이 섞일 수 있으므로 “llama.cpp는 SM90 binary다”처럼 단일 target으로 요약하지 않는다. [llama.cpp v0.2.0 — `ggml/src/ggml-cuda/CMakeLists.txt:1-58` — default architecture list](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/CMakeLists.txt#L1-L58)

native architecture detection을 쓸 때 CMake가 얻은 문자열을 정규식으로 검증한 뒤 실제 list로 채택한다. build host에 GPU가 없거나 detection 결과가 이상한 경우를 방어한다. 이것은 container build에서 `native`를 무심코 쓰면 build host GPU만 target으로 고정될 수 있다는 위험과 연결된다. [llama.cpp v0.2.0 — `ggml/src/ggml-cuda/CMakeLists.txt:80-101` — native architecture validation](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/CMakeLists.txt#L80-L101)

CUDA option surface에는 FlashAttention compile, all-quant variant, CUDA Graph, NCCL, MMQ/cuBLAS 강제 선택 등이 있다. `GGML_CUDA_FA`가 꺼지면 compile definition으로 FA path를 제외하고, force MMQ/cuBLAS도 device code branch를 바꾼다.

역할별 고정 좌표와 판정 범위는 다음과 같다.

- NCCL option은 `find_package(NCCL)` 성공 뒤 define과 link를 추가하며 찾지 못하면 경고와 generic path가 남을 수 있다.
- [llama.cpp v0.2.0 — `ggml/CMakeLists.txt:199-213` — CUDA feature options](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/CMakeLists.txt#L199-L213) [llama.cpp v0.2.0 — `ggml/src/ggml-cuda/CMakeLists.txt:115-191` — compile definitions와 NCCL gate](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/CMakeLists.txt#L115-L191)

option→artifact 검증에서 CMake cache만 보지 않는다. 실제 compile definition, target list, linked NCCL와 output library digest를 남긴다. runtime에서는 GPU backend enumeration, model layer offload와 selected operator를 따로 본다. build에 FA가 있어도 shape·dtype가 일반 kernel로 갈 수 있고, NCCL이 link돼도 single-GPU request에서는 collective를 쓰지 않는다.

**artifact journey가 미확정일 때 지켜야 할 말.**

archive member list는 wheel에 어떤 filename이 있는지 증명한다. ELF dynamic section은 host dependency를 증명한다. `.nv_fatbin` section 존재는 embedded container가 있음을 보여 준다. `cuobjdump` code-object list는 target inventory를 더 직접적으로 보여 줄 수 있고, `nvdisasm`은 cubin instruction을 읽는 데 쓰인다. 각 도구가 증명하는 범위를 바꾸지 않는다.

도구가 없는 보존 표본에서 `.nv_fatbin` filename과 `libcudart.so.13` dependency만 확인했다면 “CUDA 13 runtime dependency와 fatbin container가 있다”까지 쓴다. `sm_100` cubin과 `compute_90` PTX가 들어 있다고 쓰지 않는다. filename에 `sm90` token이 있어도 symbol별 target과 PTX fallback은 미확정이다.

반대로 inventory를 찾지 못했다고 GPU code가 없다고 단정하지 않는다. binary가 압축·embedded되거나 custom container에 있을 수 있다. 검사 도구와 method의 coverage를 기록한다. 부재 주장은 대상 artifact 전체와 relevant section을 검사했다는 증거가 필요하다.

release audit 표에는 `proven`, `contradicted`, `unknown` 세 상태를 쓴다. unknown을 false로 바꾸면 새로운 GPU 지원을 과소평가하고, true로 바꾸면 production에서 no-image를 맞는다. 미확정은 추가 binary audit 또는 canary가 필요한 명시적 작업 상태다.

## 44.7 여섯 장애의 first divergence를 찾는다

오류 문자열은 시작 좌표이지 root cause가 아니다. 같은 package에서도 kernel family와 shape마다 AOT cubin, PTX JIT, runtime source JIT 경로가 다를 수 있다. 사건마다 “정상 환경과 처음 달라진 칸”을 여덟 칸 ledger에서 찾는다.

### 44.7.1 사건 1: `no kernel image`가 첫 긴 요청에서만 났다

서버 startup과 짧은 prompt는 정상이었지만 긴 context가 split-KV backend를 선택하자 `no kernel image`가 났다. 첫 가설은 driver가 너무 오래됐다는 것이었다. driver upgrade를 준비했지만, 같은 node에서 다른 CUDA extension은 정상 실행됐다.

first divergence는 request shape→kernel symbol이었다. 짧은 prompt는 일반 attention symbol을 썼고 wheel에 device SM용 native cubin이 있었다. 긴 prompt가 선택한 split specialization은 다른 sidecar cubin package에서 제공됐는데 설치가 빠져 있었다. embedded PTX도 그 symbol에는 없었다. process import가 성공한 이유와 첫 긴 request가 실패한 이유가 동시에 설명됐다.

driver-old 가설은 동일 driver에서 일반 symbol의 native image가 실행되고, 실패 symbol inventory에 compatible cubin/PTX가 없다는 증거로 기각됐다. driver를 올려도 input PTX가 없으면 JIT할 수 없다. 수정은 올바른 cubin extra를 설치하거나 지원 backend로 dispatch하는 것이었다. wheel filename과 digest, member/manifest key를 deployment manifest에 고정했다.

복구 종료 조건은 startup 성공이 아니다. 짧은/긴, full/tail shape가 선택하는 모든 production symbol에 대해 native/compatible cubin 또는 검증된 PTX/JIT/fallback 경로가 있어야 한다. 새 GPU canary에서는 backend별 첫 launch와 selected image/rejection reason을 기록한다. health endpoint가 kernel coverage를 대신하지 않는다.

**사건 2: native image가 없어 PTX로 갔지만 driver JIT가 거부했다.**

새 GPU에서 native cubin이 없어 embedded PTX가 fallback이 됐다. package inventory에는 PTX가 있었으므로 팀은 forward compatibility가 확보됐다고 보았다. 첫 launch에서 “PTX was compiled with an unsupported toolchain” 계열 오류가 났다. system에 최신 nvcc를 설치했지만 현상은 그대로였다.

driver JIT는 system nvcc를 호출하는 경로가 아니다. installed driver의 JIT compiler가 embedded PTX를 읽는다. package에 들어 있는 PTX version/feature가 driver capability보다 새로우면 system nvcc 교체는 이미 build된 PTX를 바꾸지 않는다. first divergence는 `embedded PTX → host driver JIT`였다.

두 record를 분리했다. build host의 nvcc/toolkit과 deployment host의 driver다. `nvidia-smi`가 표시한 지원 CUDA label, Python `torch.version.cuda`, `/usr/local/cuda/bin/nvcc`도 각각 다른 값이었다. 실제 오류 판단에는 embedded PTX와 driver가 직접적이었다. compatibility document도 PTX JIT가 필요한 application은 minor compatibility의 제한을 받는다고 경고한다.

수정 후보는 driver upgrade, older PTX/native target을 포함한 rebuild, 검증된 compat package와 native cubin 배포였다. system nvcc symlink만 바꾸는 조치는 제외했다. rebuild에서는 target GPU용 real code를 포함해 older driver의 PTX JIT 의존을 줄였다. 그러나 새 feature가 driver API를 요구하면 native code만으로 해결되지 않을 수 있어 feature gate도 확인했다.

종료 조건은 한 kernel JIT 성공이 아니다. wheel의 PTX-only symbol 목록, target GPU와 driver matrix를 만들고 cold cache에서 JIT 가능한지 또는 native image가 선택되는지 증명한다. JIT cache를 지운 대조와 persistent cache hit도 나눠 첫 request latency를 확인한다.

### 44.7.2 사건 3: `undefined symbol`은 GPU image 문제가 아니었다

extension import 중 `undefined symbol`이 발생했다. package 이름에는 올바른 CUDA variant가 들어 있어 팀은 cubin target을 의심했다. 그러나 error는 kernel launch 전 dynamic loader가 host `.so` relocation을 해결하는 단계에서 났다. PTX·SASS inventory는 아직 소비되지 않았다.

first divergence는 wheel `.so`의 `NEEDED`/symbol version과 node에 load된 runtime library였다. 다른 package가 앞선 RPATH 또는 `LD_LIBRARY_PATH`로 예상과 다른 `libcudart`·PyTorch library를 가져왔다. host ABI symbol이 없어 import가 실패했다. `no kernel image`와 같은 compatibility 표에 넣으면 조사 순서를 거꾸로 탄다.

native image 가설은 GPU를 숨긴 상태에서도 import가 같은 symbol에서 실패하고, loader trace가 host library resolution에서 끊긴 것으로 기각됐다. 수정은 wheel dependency와 RPATH, container library set을 일관되게 만들고 충돌하는 host library를 제거하는 것이었다. compat user-mode driver library와 toolkit runtime library도 같은 것으로 취급하지 않았다.

복구는 extension import, expected `.so` resolution과 symbol version을 먼저 확인한다. 그다음 실제 device에서 code image selection을 별도로 검증한다. import가 성공했다고 GPU target 문제가 해결됐다는 뜻은 아니며, host dependency gate와 device code gate를 순서대로 통과했을 뿐이다.

**사건 4: host load는 됐지만 CUDA 13 backend가 조용히 fallback했다.**

새 CUDA 13 package로 올린 뒤 오류는 없었지만 attention latency가 늘었다. log에는 generic backend만 보였고 팀은 새 compiler가 느려졌다고 추측했다. 실제로는 기대한 FA backend가 import 또는 device capability predicate에서 거부돼 fallback이 선택됐다.

first divergence ledger는 option→version source부터 시작했다. source build가 아니라 hosted wheel을 설치했고 package mapping은 cu130이었다. 하지만 FA extension extra가 설치되지 않았거나 해당 device가 runtime capability gate 범위 밖이었다. CUDA 13이라는 package label은 backend selection predicate를 참으로 만들지 않았다.

compiler regression 가설은 selected kernel symbol이 이전과 다르고 expected backend가 한 번도 launch되지 않았다는 사실로 기각됐다. performance A/B는 같은 algorithm/kernel이 아니었다. 먼저 unavailable reason을 해결하거나 fallback을 명시적 baseline으로 다시 정의해야 toolkit compiler 비교가 가능하다.

수정은 backend availability를 readiness에 포함하되 무조건 process를 실패시키지는 않는 policy였다. correctness fallback을 허용하는 service는 selected backend와 SLO를 경보한다. 특정 backend가 계약인 deployment는 import, device SM, dtype/shape와 code artifact availability가 모두 맞지 않으면 traffic을 받지 않는다.

종료 조건은 expected backend 비율, fallback reason cardinality, representative shape의 symbol과 latency다. package version만 dashboard에 남기지 않는다. backend가 바뀐 성능 결과를 CUDA 12↔13 toolchain 효과라고 보고하지 않는다.

### 44.7.3 사건 5: 매 pod 첫 요청이 JIT compile로 지연됐다

새 deployment는 첫 요청 TTFT가 수십 초였고 이후 요청은 빨랐다. pod를 재시작하면 반복됐다. GPU idle 구간과 CPU compile activity가 겹쳐 JIT storm이 의심됐다. 그러나 단순히 “Triton warm-up”이라고 부르면 cache가 왜 재사용되지 않는지 알 수 없다.

first divergence는 kernel key lookup→local artifact hit였다. cache directory가 container writable layer에 있어 pod restart 때 사라졌다. 여러 replica가 같은 model/shape를 각각 compile했다. 일부 key에는 absolute source temp path가 들어가 replica마다 digest가 달라졌다. published cubin manifest miss 뒤 JIT로 간다는 사실과 JIT cache persistence는 별도 단계였다.

GPU 성능 가설은 compile 전 CPU timeline과 artifact publish 뒤 동일 symbol의 낮은 latency로 기각됐다. download 문제인지 compiler 문제인지도 나눴다. manifest lookup, remote fetch, checksum, compile start/end, module load 사건을 각각 기록했다. network retry가 긴 case와 actual compiler가 긴 case가 섞여 있었다.

수정은 cache key를 source/version/target/template에 안정적으로 묶고 cache volume을 release 단위로 보존하며, artifact를 image build 또는 controlled warm-up에서 생성하는 것이었다. 다른 driver·toolkit·SM artifact를 억지로 공유하지 않도록 key에 compatibility 입력을 남겼다. file lock과 atomic publish로 여러 pod가 incomplete file을 읽지 않게 했다.

복구 종료 조건은 warm pod 한 개가 빠른 것이 아니다. cold deployment에서 compile 횟수가 expected unique key 수를 넘지 않고, restart·replica 추가 때 cache hit하며, artifact digest와 target이 device에 맞아야 한다. stale artifact를 잘못 재사용해 빠른 대신 오답·load failure가 나지 않는지 검증한다.

### 44.7.4 사건 6: CUDA 13으로 올렸는데 extension이 제외됐다

build host를 CUDA 13으로 바꾼 뒤 오히려 optional extension이 wheel에서 사라졌다. `nvcc -V`는 13.x였으므로 version minimum predicate는 통과해야 했다. 팀은 setup bug를 의심했다. 하지만 build log와 package dependency를 대조하자 `torch.version.cuda` major가 CUDA 12인 PyTorch wheel과 build toolkit CUDA 13이 섞여 있었다.

extension별 predicate가 읽는 version source가 달랐다. setup의 FA/FlashMLA gate는 `CUDA_HOME` nvcc를 읽었지만 dependency/exclusion logic 일부는 PyTorch build variant를 읽었다. CMake compiler와 linked torch CUDA libraries가 혼합돼 extension이 안전하게 제외되거나 configure 단계에서 실패했다. `CUDA_VERSION=13` 한 줄로는 이 모순이 보이지 않았다.

“minimum version bug” 가설은 nvcc predicate 자체는 참이고 다른 dependency branch에서 제외된 source trace로 기각됐다. 수정은 supported build matrix의 PyTorch CUDA variant와 hosted toolkit을 맞추고, package mapping이 기대한 cu130 artifact를 사용하도록 하는 것이었다. source build를 원하면 compiler, torch, linked runtime과 target list를 manifest로 고정했다.

종료 조건은 wheel build success가 아니다. expected extension member/symbol, host `NEEDED`, target image/PTX inventory, import와 runtime backend 선택까지 닫는다. CUDA 13 compiler가 source를 받아들였다는 사실만으로 CUDA 13 distribution이 완성됐다고 하지 않는다.

## 44.8 진단 기록과 migration workbook

### 44.8.1 첫 15분에 수집할 좌표

첫 기록은 host driver와 visible GPU/SM이다. 둘째는 toolkit/nvcc path와 version이며 없으면 없다고 적는다. 셋째는 `torch.version.cuda` 같은 framework build variant다. 넷째는 package filename, version과 digest다. 다섯째는 host `.so`의 `NEEDED`, RPATH/RUNPATH다. 여섯째는 확인 가능한 cubin/PTX/fatbin target inventory다. 일곱째는 selected backend·symbol과 rejection reason, 여덟째는 JIT cache key/path/hit다.

이 목록은 명령어를 외우기 위한 checklist가 아니다. 오류 단계에 따라 필요한 앞뒤 좌표를 고르는 사건 지도다. import undefined symbol이면 host ELF부터 보고, first launch no-image이면 device SM과 code inventory를 보고, unsupported PTX이면 embedded PTX와 driver를 본다. silent fallback이면 framework predicate부터 본다. first-request delay면 manifest/JIT cache timeline을 본다.

**요청 한 건의 artifact 선택 timeline.**

새 GPU의 첫 decode 요청 R을 따라가자. R은 API ingress와 scheduler를 지나 attention backend selector에 도달한다. selector는 device capability, dtype, head dimension과 package import 상태를 보고 backend B를 고른다. 여기까지는 CUDA driver가 B의 모든 kernel image를 실행해 본 것이 아니다. B가 launcher에서 shape specialization K를 정할 때 비로소 정확한 symbol/key가 나온다.

K가 extension `.so`에 embedded돼 있다면 host module은 이미 import됐을 수 있지만 device code module load는 lazy일 수 있다. sidecar cubin이면 manifest key→path→checksum→module load가 추가된다. published artifact가 없으면 JIT source와 compiler가 필요하다. timeline을 다음처럼 적는다.

```text
t0 backend B availability predicate
t1 request shape selects specialization K
t2 artifact key lookup
t3 local/package/download/JIT branch
t4 host dependency and module load
t5 driver native-image selection or PTX JIT
t6 kernel symbol resolution
t7 first launch
t8 cache/artifact publish for next request
```

`no kernel image`가 t7에서 보였어도 first divergence는 t2 inventory일 수 있다. unsupported PTX는 t5다. checksum mismatch는 t3이다. undefined host symbol은 t4이고 backend fallback은 t0 이전이다. first request compile latency는 t3→t5 또는 t8 lock까지다. 오류 시각과 원인 생성 시각을 구분한다.

R 다음 요청 S가 같은 shape인데 빨라졌다면 JIT/cache 가설이 강해진다. 그러나 S가 다른 worker process로 갔으면 process-local module cache를 공유하지 않을 수 있다. same pod, same process, same artifact key인지 확인한다. dynamic batching이 K의 template parameter를 바꾸면 겉보기 같은 model도 다른 cache key를 만든다.

request cancellation도 artifact build를 자동 취소하지 않을 수 있다. R이 timeout돼도 compiler와 cache publish가 끝나 S가 hit할 수 있다. 첫 요청 실패와 두 번째 성공을 nondeterministic driver로 해석하기 전에 background compile lifetime을 본다. incomplete temp file을 S가 읽었다면 checksum·atomic rename 계약이 필요하다.

이 timeline의 관측에는 privacy-sensitive prompt 내용이 필요 없다. model/backend digest, shape bucket, device SM, artifact key digest, branch와 duration만으로 충분하다. raw request id를 global metric label로 넣지 않고 trace correlation에 제한한다.

**build manifest를 한 행씩 검증한다.**

release manifest의 한 행은 package 하나가 아니라 backend artifact 하나를 나타내야 한다. 예를 들어 `vLLM FA3, cu130, sm90, bf16, head-dim 128` 같은 행에 source commit, build toolkit/nvcc, PyTorch CUDA variant, target flags, output member/digest, host `NEEDED`, runtime device predicate와 fallback을 둔다.

첫 검증은 version source다. setup function이 `CUDA_HOME` nvcc를 읽는데 manifest가 `torch.version.cuda`만 기록했다면 build predicate를 재현할 수 없다. SGLang CMake `CUDA_VERSION`과 runtime `torch.version.cuda` major도 다른 열이다. FlashInfer system ptxas patch는 JIT compiler path 열에 들어간다.

둘째는 predicate와 artifact다. `nvcc>=12.9`라 FlashMLA build gate가 참이어도 output member가 없으면 compile/package 단계에서 빠진 것이다. CMake option이 ON이어도 source glob이 비었거나 architecture compile이 실패했을 수 있다. 반대로 extra wheel에 cubin member가 있어도 manifest key가 runtime op를 가리키지 않으면 unused artifact다.

셋째는 artifact와 runtime selection이다. target cubin이 있어도 device gate가 false이면 선택되지 않는다. backend가 선택돼도 request shape가 다른 specialization을 요구할 수 있다. representative production shape별 key를 manifest 행에 연결한다. “import test passed”를 selection proof로 쓰지 않는다.

넷째는 host dependency다. `.so`의 required runtime/library와 container에 resolved path를 기록한다. package manager dependency 선언은 실제 dynamic loader 결과와 다를 수 있다. compat package가 있다면 어느 user-mode driver library가 load됐는지 별도 열에 둔다.

다섯째는 unknown 처리다. binary tool이 없어 code-object inventory를 못 읽었다면 target을 package filename에서 추정하지 않는다. canary first launch로 일부 symbol 실행은 증명할 수 있지만 전체 inventory는 여전히 unknown이다. 다음 release gate에 binary audit tool을 추가할 작업으로 남긴다.

### 44.8.2 CUDA 12.9.1→13.3.0 migration을 통제하는 법

toolkit migration 성능 비교에서는 같은 source commit, framework와 dependency, target architecture list, compile mode와 workload를 유지한다. 12.9.1 build와 13.3.0 build의 compiler·bundled headers/library만 바꾸는 것이 이상적이다. 실제 hosted wheel은 PyTorch, FlashAttention, Triton과 package content도 다를 수 있으므로 그런 비교를 “wheel bundle A/B”라고 부른다.

각 build에서 compile command와 target flags를 보존한다. CUDA 13이 새 family target을 지원해 target list가 달라지면 같은 compiler optimization 비교가 아니다. native image/JIT 비율도 달라질 수 있다. ptxas register·spill report, cubin digest와 selected symbol을 함께 둔다.

startup·first request·warm steady state를 분리한다. CUDA 13 build에서 first request만 느리면 JIT cache와 target miss를 본다. warm kernel duration이 달라졌다면 generated code와 algorithm dispatch를 본다. package import부터 실패하면 host library/ABI를 본다. 세 구간을 평균 TTFT 하나로 합치지 않는다.

correctness 차이가 나면 compiler bug부터 선언하지 않는다. latent race 또는 undefined behavior가 scheduling 변화로 드러났을 수 있다. stream/event dependency, out-of-bounds와 natural alignment 계약을 먼저 감사한다. explicit protocol을 넣은 뒤 두 toolchain에서 안정되는지, first divergent kernel과 instruction이 무엇인지 본다.

driver도 통제한다. 동일 host driver가 두 application을 지원할 수 있는 범위라면 고정하고, 그렇지 않아 driver까지 바뀌면 결과를 toolkit+driver bundle 비교로 표시한다. compat package를 한쪽에만 쓰면 user-mode library path도 차이다. `nvidia-smi` label만 기록하지 않는다.

migration 완료는 build success가 아니다. production symbol set의 artifact coverage, cold/warm load, backend selection, output, latency·memory와 fallback rate가 기준을 통과해야 한다. rollback은 옛 wheel filename만 남기는 것이 아니라 dependency/container, JIT cache namespace와 config를 함께 되돌릴 수 있어야 한다.

### 44.8.3 rollback이 실제로 돌아가는지 증명한다

CUDA package rollback은 Python wheel 하나를 downgrade하는 일이 아니다. 새 wheel과 함께 container의 PyTorch CUDA variant, `libcudart`, driver compat library, attention extra와 JIT cache schema가 바뀌었을 수 있다. 옛 wheel만 되돌리면 새 host library를 load하거나 새 artifact cache를 읽어 또 다른 혼합 상태가 된다.

rollback bundle에는 application image digest, 모든 CUDA-related distribution filename/digest, environment/build option, selected library path와 JIT cache namespace를 포함한다. driver는 host-wide state라 즉시 되돌리기 어려울 수 있으므로 새 driver가 옛 bundle을 backward-compatible하게 실행하는지 사전 검증한다. driver rollback이 필요한 경우 traffic drain과 node lifecycle이 별도 작업이다.

첫 rollback fixture는 import다. 옛 extension이 기대한 host `.so`와 symbol을 resolve하는지 확인한다. 둘째는 code selection이다. production GPU별 representative native symbol과 PTX/JIT symbol을 cold/warm 상태에서 확인한다. 셋째는 dispatcher다. preferred/fallback backend 비율이 이전 기준으로 돌아오는지 본다. 넷째는 model output과 SLO다.

JIT cache는 가장 위험한 공유 상태다. 새 compiler가 같은 filename으로 다른 code를 만들거나 key schema가 바뀌면 옛 runtime이 incompatible artifact를 읽을 수 있다. release·toolkit·target·source digest namespace를 분리하고 rollback에서 옛 namespace를 read-only로 복원하거나 안전하게 재생성한다. cache directory 전체를 무조건 지우면 correctness 위험은 낮출 수 있지만 cold-start storm이 생기므로 capacity와 traffic 계획이 필요하다.

sidecar cubin download cache도 manifest version과 checksum으로 묶는다. rollback package가 옛 manifest를 읽는데 local path에는 새 cubin이 있으면 filename collision이 생길 수 있다. checksum mismatch가 명확히 실패하고 atomic download가 새 path를 만들도록 한다. integrity error를 무시하고 기존 file을 load하는 방식으로 availability를 얻지 않는다.

rollback canary는 정상 short request만 쓰지 않는다. 이전 장애의 first-divergence shape를 포함한다. 긴 split specialization, rare quant dtype, partial page와 multi-GPU NCCL path가 각각 옛 preferred/fallback을 선택해야 한다. production key set에서 새 release가 추가한 key는 옛 bundle이 hard reject할 수 있으므로 traffic/model config도 함께 되돌린다.

종료 조건은 “옛 version string이 보인다”가 아니다. host library, device image/JIT, backend selection과 output timeline이 이전 evidence와 일치해야 한다. 새 driver 때문에 SASS/JIT 결과가 달라질 수 있으면 bundle은 완전 동일 재현이 아니라 validated rollback path라고 표시한다.

### 44.8.4 세 가지 그럴듯하지만 틀린 설명

첫째, “`nvidia-smi`에 CUDA 13이라고 나오므로 CUDA 13 wheel이 실행된다.” 이 표시는 driver capability의 한 표현이지 wheel target inventory와 host dependency를 증명하지 않는다. 둘째, “nvcc를 업그레이드했으므로 installed extension도 CUDA 13이 됐다.” prebuilt artifact는 재빌드되지 않는다. 셋째, “PTX가 있으므로 모든 미래 GPU를 지원한다.” driver JIT, PTX target·feature와 framework gate가 남는다.

반대로 old nvcc가 host에 없다는 사실만으로 prebuilt cubin 실행 실패를 예측할 수도 없다. deployment에 compiler가 필요 없는 AOT path가 있다. source JIT가 필요한 kernel만 compiler availability를 요구한다. compiler installation은 package별 runtime requirement다.

넷째 오해는 “cubin이 하나 있으니 그 extension은 AOT다”다. 같은 `.so` 안에서 일반 kernel은 cubin, rare specialization은 PTX 또는 runtime JIT source일 수 있다. AOT/JIT 분류를 package 단위가 아니라 requested symbol/key 단위로 한다. 첫 요청만 느린 현상은 extension 전체가 JIT라는 증거가 아니며, 그 key가 cache miss였다는 증거부터 찾는다.

다섯째는 “SASS가 같으면 성능도 같다”다. 같은 instruction sequence라도 launch shape, address, cache state, clocks와 concurrent workload가 다르면 duration은 달라진다. SASS equality는 compiler-code hypothesis를 좁히는 증거이지 runtime state를 고정하지 않는다. 반대로 SASS가 달라도 algorithm과 resource 결과가 같을 수 있다. instruction diff와 serving metric을 직접 인과로 붙이지 않는다.

여섯째는 “fatbin에 PTX가 있으니 package size 낭비다”다. PTX는 native image가 없는 새 device에서 JIT fallback을 제공할 수 있고, 여러 real target cubin은 cold-start JIT를 피한다. 어느 조합이 적절한지는 배포 GPU fleet, driver policy, first-request SLO, package size와 offline environment에 달려 있다. 모든 target을 native로 싣거나 PTX-only로 만드는 단일 정답은 없다.

일곱째는 “driver가 새로우면 undefined symbol도 해결된다”다. host `.so`가 기대한 framework/runtime ABI symbol은 dynamic loader와 installed user-space library의 문제일 수 있다. driver upgrade가 해당 `.so`를 제공하지 않는다. error가 relocation인지 driver API return인지 먼저 분류한다.

여덟째는 “compat package를 넣으면 driver upgrade와 같다”다. forward-compat package는 공식 지원 조건에서 특정 user-mode driver library를 제공하는 방식이며 host kernel driver 전체를 바꾸지 않는다. 지원 matrix 밖의 GPU나 feature를 임의로 살리지 않는다. load path가 compat library를 실제 선택했는지도 증명해야 한다.

아홉째는 “fallback이면 안전하므로 관측할 필요가 없다”다. correctness fallback이 service availability를 지켜도 latency, workspace, KV layout, graph capture와 numeric path가 달라진다. fallback rate가 traffic shape나 특정 tenant와 연결되면 p99만 악화될 수 있다. preferred backend rejection reason과 fallback SLO를 함께 둔다.

열째는 “build log에 `sm_90`이 보이면 wheel에 들어갔다”다. compile command가 실행돼도 link/package 단계에서 object가 빠지거나 strip·selection이 달라질 수 있다. build flag, linker output, wheel member와 binary inventory를 연속 증거로 잇는다. source target과 shipping target은 같은 칸이 아니다.

이 오해들을 반증하는 공통 질문은 하나다. 지금 말하는 version·artifact·성공이 여덟 단계 중 어느 칸의 사실인가. 그 칸의 앞 입력과 뒤 consumer를 연결할 수 없다면 결론을 한 단계 낮춘다. 예를 들어 “nvcc 13.0이 target을 받아들였다”는 build predicate 증거이고, “배포 wheel이 새 GPU에서 K를 실행한다”는 artifact와 runtime까지 필요한 별도 주장이다.

좋은 장애 보고서는 부정확한 종합 label을 지운다. `CUDA mismatch` 대신 `host ELF dependency`, `native image absent`, `PTX JIT rejected`, `runtime device gate`, `JIT cache miss` 중 관측된 first divergence를 쓴다. 아직 inventory를 못 읽었으면 `native/PTX coverage unknown`이라고 쓴다. unknown을 견디는 표현이 조급한 오진보다 빠른 다음 조사를 만든다.

release acceptance는 네 묶음의 증거로 닫는다. 첫 묶음은 공급 artifact다. source commit, build option과 version source, target flags, package·member digest와 host dependency가 서로 이어져야 한다. 두 번째는 device coverage다. fleet의 SM과 production symbol/key마다 native, validated PTX/JIT, validated fallback 또는 reject를 표시한다. file-level PTX 존재를 symbol-level coverage로 올리지 않는다.

세 번째는 시간과 correctness다. import, cold first launch, warm launch를 나누고 JIT/download/cache publish 시간을 기록한다. representative full·tail·rare dtype에서 output과 selected backend를 확인한다. native image 성공 한 건으로 JIT-only tail을 덮지 않는다. 네 번째는 운영 수명이다. replica restart와 node 교체 뒤 cache hit·integrity가 유지되고 rollback bundle이 옛 dependency·artifact namespace를 안전하게 복원해야 한다.

이 네 묶음 가운데 하나가 unknown이면 release 자체를 무조건 막으라는 뜻은 아니다. 해당 key를 dispatcher에서 reject하거나 validated fallback으로 보내고 traffic coverage를 제한할 수 있다. 중요한 것은 unknown이 preferred production path에 조용히 들어오지 않는 것이다. rollout percentage와 GPU pool을 artifact coverage에 맞춰 나눈다.

예를 들어 새 SM pool에서는 generic attention만 validated PTX이고 rare quant kernel은 unknown이라고 하자. model/tenant routing으로 rare quant traffic을 옛 pool에 남기고 generic traffic만 canary할 수 있다. 이후 binary audit 또는 first-launch evidence가 닫히면 coverage를 넓힌다. “새 GPU 지원”이라는 전체 on/off flag보다 안전하고 설명 가능한 배포다.

회귀 fixture는 장애에서 발견한 first divergence를 그대로 보존한다. no-image 사건의 긴 split key, unsupported PTX 사건의 cold JIT, undefined symbol 사건의 resolved library path, fallback 사건의 rejection reason, JIT storm 사건의 restart cache hit, mixed CUDA 사건의 nvcc·torch version source를 각각 release gate에 넣는다. 일반 smoke test를 늘리는 것보다 재발 조건을 정확히 고정한다.

마지막 승인 문장은 이렇게 쓴다. “cu130 bundle X는 driver 580+의 SM90 pool에서 production key 98%를 native로, 2%를 validated JIT로 처리하며 fallback 0이다. cold JIT artifact는 persistent namespace Y에 checksum Z로 publish되고 restart hit가 검증됐다. SM100 pool은 backend Q가 device gate에서 reject되므로 traffic 대상이 아니다.” 이 문장에는 version 숫자보다 실제 선택 경로가 많다. 그래서 다음 release diff에서도 다시 계산할 수 있다.

승인 뒤에도 fleet 구성이 바뀌면 coverage를 다시 계산한다. 같은 image를 새 GPU node pool에 배치하는 것은 단순한 horizontal scaling이 아니다. device SM이 달라져 native/PTX/JIT 후보와 dispatcher가 변한다. driver rollout도 PTX JIT compiler와 user-mode library를 바꿀 수 있다. 따라서 node image·driver·GPU SKU 변경은 application release가 없어도 artifact-compatibility canary를 거친다. 배포 도구가 container digest만 비교해 unchanged라고 판단하지 않게 한다.

반대로 application wheel만 바뀌었는데 GPU pool과 driver가 같다면 target inventory와 dependency diff에 집중할 수 있다. 모든 matrix를 처음부터 반복하는 대신 changed predicate와 affected production key를 계산한다. 이러한 incremental audit가 가능하려면 이전 release의 version source·target·symbol·fallback record가 남아 있어야 한다.

**마지막 연결: 버전 숫자 대신 변환과 선택을 기록한다.**

PTX, SASS, cubin, fatbinary와 wheel은 한 물건의 다른 이름이 아니다. PTX는 virtual ISA 입력이고, cubin은 real target code object이며, SASS는 machine instruction 표현이다. fatbinary는 여러 GPU image를 담는 container이고 wheel은 host·Python·GPU artifact를 운반하는 package다. driver는 device에 맞는 image를 선택하거나 PTX를 JIT한다.

CUDA 12.x와 13.x 비교도 한 숫자로 끝나지 않는다. build nvcc/toolkit, PyTorch build variant, bundled cudart, host driver, compat package, device SM과 target inventory를 나눈다. minor compatibility, backward compatibility와 forward-compat package는 서로 다른 방향과 조건을 가진다. PTX JIT가 필요한 순간에는 native cubin만 쓰는 application과 다른 driver 조건이 드러난다.

vLLM·SGLang·FlashInfer·llama.cpp source는 지원이 연쇄라는 사실을 보여 줬다. version source를 읽는 build predicate, emitted target, package placement, import·device gate와 runtime offload가 모두 이어져야 한다. 어느 한 칸만 보고 “지원”이라고 쓰면 startup 성공 뒤 첫 shape 실패나 silent fallback을 놓친다.

가장 실용적인 문장은 “CUDA mismatch”가 아니다. “cu130 wheel의 generic backend는 load됐지만 SM X용 split specialization에 native image와 PTX가 없었고, context threshold에서 처음 그 symbol을 선택해 no-image가 발생했다”처럼 first divergence를 적는다. 이 문장이 수정과 회귀 fixture를 직접 가리킨다.

다음 장에서는 올바른 binary가 선택됐다는 전제 위에서 FlashAttention과 FlashInfer가 실제 attention tile과 online softmax를 어떻게 구성하는지 본다. 44장이 ‘어떤 code가 GPU에 도달했는가’를 닫았다면 45장은 ‘그 code가 어떤 algorithm과 memory schedule을 실행하는가’를 닫는다.

## 44.9 CUDA 12.x→13.x 공식 차이를 판정 좌표로 읽는다

CUDA major 숫자 하나를 호환성 결론으로 쓰지 않는다. NVIDIA 공식 release notes와 compatibility 문서는 toolkit compiler/runtime, minimum driver, minor-version compatibility 범위, forward-compatibility package, deprecated/removed targets와 host platform 지원을 서로 다른 표로 제공한다. migration workbook도 같은 열을 갖는다.

첫 열은 build toolkit이다. `nvcc`와 `ptxas` 버전, target compute capabilities, emitted PTX ISA/cubin과 host compiler support를 기록한다. CUDA 13.x로 빌드했다는 사실은 CUDA 12.x-built extension을 자동 재빌드하지 않으며, installed nvcc는 wheel 내부 artifact를 바꾸지 않는다.

둘째 열은 host driver다. driver가 application/toolkit에서 요구하는 runtime/driver API와 PTX JIT input을 지원하는지 공식 compatibility table로 판정한다. “driver version 숫자가 toolkit보다 높다”는 문자열 비교 대신 release family와 documented minimum/range를 사용한다.

셋째 열은 runtime libraries다. cudart, cuBLAS, cuDNN, NCCL와 other DSOs가 wheel/container/system 중 어디서 오는지 적는다. toolkit compiler와 runtime DSO version은 같은 coordinate가 아니다. loader path가 예상과 다른 `libcudart` 또는 framework library를 고를 수 있다.

넷째 열은 device SM이다. GPU product name보다 compute capability와 feature requirements를 사용한다. `sm_90`, `sm_90a`처럼 architecture-specific feature target은 일반 SM 숫자 비교로 호환을 단정하지 않는다. official compiler target support와 code-object inventory를 확인한다.

다섯째 열은 native cubin coverage다. wheel/fatbinary에 production specialization symbol/key가 device와 compatible한 real target으로 있는지 본다. filename의 `cu12`/`cu13`, package metadata나 member count는 native image 목록의 대체물이 아니다.

여섯째 열은 PTX fallback이다. embedded PTX가 있는지, virtual target와 PTX ISA가 무엇인지, driver JIT가 지원하는지 확인한다. PTX 존재가 모든 future GPU 지원을 무조건 보장하지 않는다. target features와 driver JIT capability, application compatibility 조건이 있다.

일곱째 열은 host ABI다. Python ABI/tag, PyTorch C++/CUDA extension ABI, libstdc++ ABI, framework/exported symbols와 dependent DSO SONAME/resolution을 확인한다. `undefined symbol`은 SM/cubin/PTX 문제가 아니라 host loader/symbol contract일 수 있다.

여덟째 열은 runtime dispatcher다. vLLM/SGLang/FlashInfer/attention backend가 device, dtype, head dimension, page size와 feature option으로 어느 extension/symbol을 선택하는지 본다. compatible image가 있어도 gate가 reject해 fallback할 수 있고, startup import가 성공해도 첫 rare key에서 image가 없을 수 있다.

CUDA 12.x→13.x diff는 이 열마다 official fact와 project-specific inference를 분리한다. official 문서가 minimum driver/target deprecation을 말하면 사실이다. 특정 vLLM wheel이 해당 target을 싣는지는 artifact/source 증거다. “따라서 우리 SM100 pod가 실패할 것”은 두 증거를 production key와 결합한 판정이다.

release notes는 전체 migration 영향의 시작점이지 project binary manifest가 아니다. removed compiler target이나 host toolchain change가 source build에 영향을 줄 수 있고, library behavior/API change는 별 component notes를 확인해야 한다. 기억으로 CUDA major 차이를 나열하지 않는다.

## 44.10 driver/toolkit/SM/PTX/cubin/host ABI decision tree

첫 질문은 Python extension import가 성공하는가다. 실패하면 loader error와 resolved dependency paths를 본다. missing DSO, GLIBCXX/libstdc++, Python/PyTorch ABI 또는 undefined framework symbol을 먼저 판정한다. GPU image inventory로 내려가지 않는다.

import가 성공하면 backend/device gate가 target extension을 선택하는가. rejection reason과 fallback을 기록한다. gate가 선택하지 않았다면 preferred kernel performance/coverage 문제이고, fallback이 검증됐는지 본다. `no kernel image`가 아직 없다는 이유로 native coverage가 있다고 말하지 않는다.

extension이 선택되면 requested production symbol/specialization에 compatible native cubin이 있는가. 있다면 driver가 load/launch할 수 있는지 본다. exact target feature와 device capability, module load errors를 확인한다. compatible native path에서는 local nvcc가 필요하지 않다.

native가 없으면 embedded PTX가 있는가. 없으면 explicit no-image/reject/fallback이다. 있으면 PTX virtual target/ISA와 driver JIT compatibility를 공식 조건으로 판정한다. persistent JIT cache와 cold/warm latency, JIT failure logs를 기록한다.

native/PTX 모두 가능해도 wrong symbol/shape가 선택될 수 있다. dispatcher key와 artifact symbol inventory를 연결한다. common short request가 generic native kernel을 쓰고 long split request만 missing specialization을 선택하면 startup/health는 통과한다.

host ABI와 device code는 독립 축이지만 extension load 과정에서 만난다. `.so`가 load되지 않으면 embedded fatbinary에 도달하지 않는다. host load 성공 후 module/kernel launch에서 device-code 문제가 드러난다. 오류 발생 층과 artifact 생성 층을 구분한다.

driver/toolkit 관계도 방향을 명시한다. newer driver가 older toolkit-built native cubin/runtime application을 지원하는 backward compatibility와, older driver에서 newer toolkit application을 제한적으로 지원하는 minor compatibility, forward-compat package 사용은 서로 다른 계약이다. 임의 숫자 부등식으로 합치지 않는다.

decision record의 leaf는 `native`, `validated PTX JIT`, `validated fallback`, `host reject`, `device gate reject`, `no image`, `unsupported PTX`, `unknown`이다. success 하나로 합치지 않는다. production key마다 leaf와 증거 artifact를 둔다.

unknown은 canary 대상 또는 admission reject다. filename/default에서 추론해 preferred path로 흘리지 않는다. binary inspection 도구가 없으면 확인하지 못했다고 남기고 controlled first-launch로 validation한다.

## 44.11 vLLM·SGLang wheel extension 로딩 경로

vLLM route에서 model/request shape가 attention/backend selector에 들어가 어떤 Python module/native extension을 import하고 어느 op/symbol을 호출하는지 걷는다. setup/build가 CUDA version source와 package variant를 정하는 것, runtime device gate가 backend를 고르는 것은 다른 단계다.

wheel 설치 시 Python package metadata와 `.so` members, bundled libraries와 GPU images가 배치된다. process import는 Python ABI와 host dependencies를 해결하고 extension initialization을 실행한다. 특정 CUDA kernel은 lazy module load/first op call에 도달할 때만 검증될 수 있다.

vLLM의 FlashAttention/FlashMLA/other extension gates는 compute capability, dtype/model feature와 build availability를 함께 볼 수 있다. gate가 false면 fallback backend가 선택될 수 있다. fallback success를 requested backend support로 기록하지 않고 selected backend reason을 trace한다.

SGLang AOT kernels는 CMake/packaging target policy와 CUDA version gates를 거쳐 wheel artifact를 만든다. runtime은 installed package/extensions와 device capability, operation key를 사용한다. AOT coverage가 없으면 JIT resolver 또는 fallback이 있는지 source에서 확인한다.

SGLang/FlashInfer JIT path는 source/package version, architecture, compiler/toolkit, flags와 kernel key를 cache identity에 포함해야 한다. container의 nvcc와 framework CUDA build mismatch가 compile/load ABI 문제를 만들 수 있다. JIT success와 persistent cache publish/next restart hit를 분리한다.

extension loading trace에는 wheel filename/digest, `.so` path/digest, resolved dependencies, framework/runtime versions, device SM, selected backend/key, native/PTX/JIT leaf, first launch와 fallback을 둔다. `torch.version.cuda`와 `nvcc --version`은 서로 다른 source다.

rolling fleet에서는 pod/container digest가 같아도 node driver/GPU가 달라 leaf가 바뀔 수 있다. 같은 wheel이 SM90에서 native, SM100에서 PTX/fallback/reject일 수 있다. GPU node pool 추가는 artifact compatibility rollout이다.

반대로 driver/GPU가 같아도 wheel update가 host ABI, target inventory와 selector를 바꾼다. source version diff만 보지 않고 actual distribution inventory와 first-launch matrix를 비교한다. build server environment가 artifact에 반영됐는지 attestation을 남긴다.

loader incident는 request timeline에 연결한다. API accepted→model/backend select→Python op→extension module/symbol→driver module/image→kernel launch→output/fallback이다. first divergence가 import, gate, image selection, JIT 또는 launch 중 어디인지 표시한다.

## 44.12 “버전 숫자가 높으면 호환된다” 오진 incident

관측은 CUDA 13 계열 toolkit이 설치된 새 node에서 vLLM process와 model load가 성공했지만 첫 long-context request가 `no kernel image`로 실패한 것이다. 운영자는 driver와 toolkit 숫자가 이전보다 높으므로 호환돼야 한다고 판단했고, framework 재시작과 nvcc path 변경을 반복했다.

첫 분기는 host load와 device launch다. Python extension import, dependent DSO resolution과 op registration은 성공했다. `undefined symbol`이나 missing library가 없다. 따라서 Python/PyTorch/host ABI가 이 요청의 first failure는 아니다. 특정 op 첫 launch까지 내려간다.

둘째 분기는 backend selector다. short-context requests는 generic backend/native image를 사용해 성공했고 long context threshold에서 split specialization을 선택했다. startup health는 long key를 실행하지 않았다. framework 전체 CUDA 지원이 아니라 one production key coverage 문제다.

셋째 분기는 artifact inventory다. wheel filename은 `cu13` variant였지만 target extension의 long specialization에는 older real targets만 있고 new device SM compatible cubin이 없었다. embedded PTX도 해당 symbol/key에는 없거나 확인되지 않았다. package label은 compiler/runtime family 표지였지 complete target coverage가 아니었다.

node에 CUDA 13 nvcc가 설치됐어도 prebuilt `.so`/fatbinary는 자동 재compile되지 않는다. runtime op는 wheel embedded artifact를 load했다. JIT source path가 없거나 disabled라면 local nvcc 변경은 no-op이다. 이 fact가 “더 높은 toolkit이면 image를 생성한다” 가설을 반증한다.

driver가 newer인 것은 older compatible native cubin 실행에 도움이 될 수 있지만 존재하지 않는 device image를 만들어 주지 않는다. PTX가 있어야 driver JIT 후보가 된다. PTX도 symbol/virtual target/ISA 조건을 만족해야 한다. 숫자가 높다는 사실은 missing artifact를 보충하지 않는다.

first divergence는 build/package target inventory→runtime selected long specialization 사이 coverage gap이다. 오류 문자열은 driver launch에서 나타났지만 원인은 wheel을 만들 때 해당 production key의 compatible native/PTX image를 싣지 않은 것이다. driver/toolkit 교체가 아니라 artifact/selector owner가 수정 대상이다.

incident evidence는 exact wheel/extension digest, device SM, driver, resolved `.so`, selected backend/specialization key, native/PTX inventory, module/launch error와 passing short key를 포함한다. binary inventory를 추출하지 못했다면 no-image error와 build manifest/source target을 근거로 조건부 판정하고 미확정 칸을 남긴다.

수정 후보는 compatible target로 extension wheel을 rebuild, validated PTX를 포함, device gate에서 unsupported key를 explicit reject하거나 verified fallback으로 route하는 것이다. generic fallback이 correctness/SLO를 만족한다면 containment가 될 수 있다. silent fallback은 관측/승인 없이 사용하지 않는다.

회귀 fixture는 exact device SM과 long threshold `K-1,K,K+1`, dtype/head dim/page/split key를 포함한다. short health와 long rare specialization을 모두 첫-launch한다. leaf가 expected native/PTX/fallback/reject인지 trace하고 output parity와 SLO를 검증한다.

host ABI fixture도 별도로 유지한다. same wheel을 expected Python/PyTorch/libstdc++/DSO matrix에서 import하고 op symbol을 resolve한다. device coverage fix가 host undefined symbol을 새로 만들지 않게 한다. import success만으로 kernel coverage를 승인하지 않는다.

rollback은 new wheel/container generation admission을 fence하고 inflight requests를 drain한다. loaded modules/JIT/graph caches와 workers를 old artifact와 혼용하지 않는다. verified previous wheel을 node SM coverage matrix에 맞는 pool로 되돌린다. previous wheel도 new SM을 지원하지 않으면 traffic routing을 old GPU pool로 제한한다.

client terminal과 already accepted long requests를 처리한다. safe retry/fallback이 가능하면 idempotency와 partial stream commit을 확인한다. 그렇지 않으면 explicit error를 준다. worker restart로 error가 사라지지 않는 missing image를 반복하지 않는다.

rollback 완료는 process health가 아니다. old artifact digest, selected backend leaf, long-key first launch, output parity, no cold-JIT/fallback surprises와 request/resource/telemetry terminal을 확인한다. node driver/GPU가 unchanged여도 artifact matrix를 다시 판정한다.

incident 문장은 “CUDA 13인데 호환되지 않았다”가 아니다. “cu13 wheel의 extension import는 성공했지만 SM X에서 long split key Y가 선택한 symbol에 compatible native cubin/PTX가 없어 first launch에서 no-image가 발생했다”라고 쓴다. version label보다 path와 missing leaf가 중요하다.

## 44.13 CUDA 12→13 migration·verification·rollback terminal

migration baseline은 source commit, wheel/container digest, build toolkit/nvcc/host compiler, PyTorch build variant, bundled DSOs, target inventory, driver/GPU pools와 production key distribution이다. candidate도 같은 schema로 수집한다. `CUDA_VERSION=13` 한 줄 diff로 끝내지 않는다.

공식 release notes diff에는 compiler/target support, minimum/supported driver compatibility, removed/deprecated components, host compiler/platform와 relevant runtime/library changes를 적는다. 각 항 옆에 official URL/version과 exact statement 범위를 둔다. project impact는 별 inference 열에 쓴다.

build phase는 clean environment에서 target matrix를 명시한다. emitted real/virtual targets와 PTX policy, framework extension ABI, RPATH/DSO bundle과 reproducible artifact digests를 manifest로 만든다. build log의 nvcc version과 resulting binary inventory를 연결한다.

distribution phase는 wheel tags, Python ABI, `.so` members, bundled libraries와 package dependency resolution을 확인한다. index resolver가 intended cu12/cu13 variant를 골랐는지 lockfile/artifact digest로 고정한다. filename이 같은 rebuild도 digest가 다르면 새 artifact다.

host-load phase는 target containers/nodes에서 Python import, `ld` dependency resolution, framework op registration과 host symbols을 검사한다. driver/library paths를 수집한다. 이 단계 failure는 GPU image launch matrix로 넘어가지 않는다.

device-selection phase는 each GPU SM, selected backend/device gate와 rejection/fallback reason을 본다. target inventory와 production key별 expected leaf를 만든다. architecture-specific targets와 generic targets를 구분한다. fleet의 최소/최대만 테스트하지 않고 distinct SM families를 모두 포함한다.

first-launch phase는 common and rare keys를 cold process/JIT cache에서 실행한다. native, PTX JIT, source JIT, fallback과 reject를 구분한다. cold latency, compile logs/artifact publish와 warm restart hit를 측정한다. startup success를 coverage proxy로 쓰지 않는다.

correctness phase는 baseline/candidate output/logit or kernel reference parity를 dtype/shape/edge key에서 검증한다. PTX/native/fallback paths가 semantic contract를 보존하는지 본다. faster load나 higher target coverage가 wrong output를 정당화하지 않는다.

performance phase는 selected leaf별 compile/startup latency, first-request TTFT, steady kernel latency/goodput와 memory/workspace를 비교한다. JIT cost를 steady state에 숨기지 않고 cold/warm을 분리한다. fallback cohort를 preferred cohort 평균에 섞지 않는다.

rolling rollout은 node GPU/driver, container/wheel과 JIT cache generation을 cohort labels로 둔다. old/new process가 same external persistent JIT cache를 쓴다면 cache key에 compiler/source/flags/SM ABI가 충분한지 확인한다. unsafe cross-generation hit를 막는다.

admission/readiness는 production keys의 expected leaf coverage와 first-launch canary를 포함한다. unknown이나 unexpected fallback은 traffic을 받지 않는다. health endpoint가 model load만 확인하면 별 kernel-coverage readiness를 둔다.

monitoring은 import/module/JIT/launch error, selected backend/leaf, fallback/reject reason, cold compile/cache hit와 key-cohort latency를 bounded labels로 둔다. wheel/device/request detail은 trace artifact에 둔다. high-cardinality specialization을 metric label로 무제한 노출하지 않는다.

rollback trigger는 host ABI load failure, no-image/unsupported PTX, unexpected fallback, JIT storm/cache mismatch, correctness parity failure, cold/steady SLO와 memory regression이다. trigger는 affected SM/key cohort를 즉시 fence한다. 전체 fleet restart보다 scope를 정확히 제한한다.

rollback bundle은 old wheel/container and dependency lock, compatible node driver/GPU routing, JIT cache namespace, graphs/modules와 readiness fixtures를 포함한다. candidate-built persistent artifacts를 old runtime이 재사용하지 않게 generation을 격리한다.

inflight terminal은 Python request/stream, extension op, CUDA module/kernel/events, JIT compiler process/temp/cache artifact와 telemetry를 닫는다. source JIT child가 남아 old artifact를 뒤늦게 publish하지 않게 generation guard를 둔다.

rollback 뒤 common/rare keys를 cold/warm 재검증한다. old artifact가 candidate driver에서 공식 compatibility 범위 안인지 다시 판정한다. “이전 버전이었으니 안전”이라고 가정하지 않는다. node driver도 rollout됐다면 application rollback만으로 baseline 좌표가 복원되지 않는다.

driver rollback은 system-level change이며 framework wheel rollback과 별 권한/절차다. forward-compat package나 container user-mode components가 있다면 resolved library set을 확인한다. reboot/restart 후 effective driver/library paths와 GPU visibility를 다시 수집한다.

최종 migration table의 rows는 GPU pool×production key이고 columns는 host load, backend gate, native target, PTX/JIT, fallback, output parity, cold/warm latency와 rollback leaf다. 모든 important rows가 native/validated-JIT/validated-fallback/explicit-reject 중 하나여야 한다.

공식 사실과 inference를 문장에서도 구분한다. “CUDA Compatibility 문서는 조건 X에서 minor compatibility를 설명한다”는 official fact다. “우리 wheel은 target inventory Y와 driver Z이므로 row R에서 native가 선택될 것으로 예상한다”는 inference다. canary trace가 native를 확인하면 observed fact가 된다.

독자 최종 checklist는 숫자 비교가 아니라 경로 질문이다. 누가 build했고 무엇을 emitted했는가. wheel이 무엇을 운반하고 host가 무엇을 load했는가. dispatcher가 어떤 symbol/key를 골랐는가. driver가 native/PTX 중 무엇을 선택했는가. 실패와 fallback은 어디서 관측됐는가.

이 답이 연결되면 CUDA 12.x→13.x migration은 막연한 upgrade가 아니라 artifact와 compatibility predicate의 controlled diff다. 다음 release에서 toolkit, driver, GPU 또는 wheel 하나만 변해도 affected rows를 다시 계산하고 안전하게 canary/rollback할 수 있다.

**host ABI decision을 실제 오류별로 연습한다.**

`ImportError: undefined symbol`이 뜨면 symbol 이름과 어느 `.so`가 참조/제공해야 하는지 찾는다. Python extension의 framework C++ symbol, CUDA runtime/driver symbol, libstdc++ symbol을 구분한다. `ldd`/loader trace와 package paths가 expected bundle을 가리키는지 확인한다.

PyTorch extension이 build 당시 framework/exported ABI와 runtime PyTorch가 다르면 CUDA device image가 완벽해도 import에서 실패할 수 있다. `torch.version.cuda`가 같다는 사실은 C++ symbols/ABI 호환 증거가 아니다. framework version/build flags와 extension build provenance를 맞춘다.

libstdc++/GLIBCXX error는 host toolchain/runtime 문제다. nvcc가 사용하는 host compiler support와 container base libraries를 official toolkit release notes/support matrix에 맞춘다. GPU driver를 올려 symbol을 해결하려 하지 않는다.

missing `libcudart.so` 또는 wrong SONAME는 wheel이 bundled runtime을 기대하는지 system DSO를 기대하는지 본다. RPATH/RUNPATH와 loader search order가 핵심이다. nvcc 설치 디렉터리가 runtime loader path와 같다고 가정하지 않는다.

`CUDA driver version is insufficient` 계열은 application/driver API가 요구하는 capability와 effective loaded driver library를 본다. container stub library를 runtime에서 잘못 load하지 않았는지 확인한다. version string보다 resolved path와 official compatibility 조건을 쓴다.

import 성공 뒤 `invalid device function`/`no kernel image`면 device-code target/selection으로 내려간다. host ABI 조사를 반복하지 않는다. 같은 `.so` 안에서도 일부 symbols/keys만 image coverage가 다를 수 있다.

PTX JIT failure면 embedded PTX version/target, driver JIT compiler와 feature requirements를 본다. local ptxas/nvcc를 바꿔도 driver JIT path가 바뀌지 않을 수 있다. source JIT path와 embedded PTX driver JIT를 분리한다.

이 분류를 regression에 넣는다. intentionally mismatched host framework fixture는 import에서 fail, missing native/with PTX는 JIT leaf, missing both는 explicit no-image/reject가 expected다. 오류가 올바른 층에서 명시적으로 발생하는 것도 계약이다.

**SM과 target 표기를 손으로 판정한다.**

build manifest에 `sm_80`, `sm_90`, `compute_90` PTX가 있다고 하자. SM80 device는 native80, SM90은 native90 후보다. 다른 newer SM은 compatible PTX/JIT 가능성을 official target/driver 조건으로 판정하고, native90이 무조건 실행된다고 단정하지 않는다.

architecture-specific suffix target은 feature contract가 다를 수 있다. generic numeric comparison으로 `90a≤100` 같은 판정을 만들지 않는다. compiler documentation과 artifact target metadata, selected source feature를 함께 본다. unsupported instruction feature가 있는 code object는 별 target이다.

fatbinary가 multiple code objects를 가졌다는 사실만으로 production symbol이 모두 각 target에 있다고 가정하지 않는다. template/filter/build conditional 때문에 symbol별 inventory가 다를 수 있다. long/split/quantized specializations을 exact key로 확인한다.

PTX virtual target도 lower가 항상 모든 feature를 표현하지 않는다. source가 newer feature를 요구하면 older compute target PTX를 만들 수 없거나 fallback algorithm을 써야 한다. generic PTX를 포함한다는 이유로 preferred feature path coverage라고 쓰지 않는다.

device capability gate가 artifact inventory보다 보수적일 수 있다. compatible image가 있어도 framework가 unvalidated SM을 reject할 수 있다. 이는 no-image와 다른 first divergence다. gate를 넓히려면 correctness/performance validation과 source predicate 변경이 필요하다.

반대로 gate가 낙관적이면 unsupported key가 launch까지 내려가 no-image가 된다. build manifest coverage를 gate generation에 입력하거나 runtime explicit check를 둔다. hardcoded SM list와 packaging target list가 release마다 drift하지 않게 한다.

**JIT cache와 cold-start 사건을 분리한다.**

embedded PTX driver JIT는 driver-managed cache를 사용할 수 있고, source JIT extension은 project/compiler/cache resolver를 사용할 수 있다. 둘의 cache key, storage와 invalidation이 다르다. “JIT cache hit” 하나로 합치지 않는다.

source JIT key에는 source digest, compiler/toolkit, target SM, flags, framework ABI와 relevant headers/dependencies가 필요하다. package filename이나 CUDA major만 쓰면 stale binary를 reuse할 수 있다. atomic publish와 concurrent builders도 본다.

cold pod 첫 request가 30초, warm이 20ms이면 compile/build/load stages를 trace한다. import delay, source generation, nvcc compile, link, module load와 graph warm-up을 분리한다. PTX driver JIT와 source build를 같은 `compile` span으로 뭉개지 않는다.

persistent cache가 pod-local ephemeral이면 restart마다 cold storm이 반복된다. shared cache를 쓰면 cross-node ABI/compiler contamination과 concurrency를 관리해야 한다. artifact attestation/checksum과 namespace generation을 둔다.

JIT failure fallback이 있으면 first request latency와 selected path를 기록한다. compile timeout 뒤 generic fallback이 성공해도 preferred coverage는 실패다. repeated retry가 request마다 compile storm을 만들지 않게 failure cache/backoff를 검토한다.

rollback은 candidate JIT namespace publication을 막고 running compiler children을 terminal한다. partial `.so`를 readers가 load하지 않도록 temp→atomic rename을 쓴다. old verified cache namespace와 binary fingerprint로 readiness를 확인한다.

**vLLM/SGLang option과 build mutation chain.**

CUDA version source가 environment, PyTorch build info 또는 nvcc 중 무엇을 읽는지 exact source를 찾는다. build isolation/container에서 값이 production assumption과 다를 수 있다. normalized version이 package suffix, dependencies와 extension inclusion gates를 어떻게 바꾸는지 걷는다.

architecture list option은 real/virtual targets로 normalize되고 CMake/nvcc flags 또는 AOT generator inventory를 바꾼다. unknown/new SM 처리, default target discovery와 cross-compile behavior를 본다. build machine GPU만 탐지해 fleet target을 누락하지 않게 한다.

extension enable/disable gates는 CUDA/toolkit version, compute capability features, optional dependency와 source availability를 조합한다. 제외된 extension은 runtime에서 import 불가 또는 backend rejection/fallback을 만든다. build warning을 release manifest에 반영한다.

runtime backend option은 constructed selector와 device capability branch를 바꾼다. option이 requested되어도 extension unavailable/capability reject로 fallback할 수 있다. selected/effective backend와 reason을 trace한다.

JIT enable/cache option은 missing AOT key의 resolver와 compiler path, cache directory/timeout를 바꾼다. production image에 nvcc/headers가 없는 경우 fallback/reject가 달라진다. binary-only deployment와 JIT-capable deployment를 별 recipe로 둔다.

각 option card는 parser/default→normalized state→build/runtime component→emitted artifact/selected branch→observable effect→falsifier→rollback을 갖는다. “CUDA 13 지원” checkbox만 만들지 않는다.

**두 migration row를 실제로 비교한다.**

baseline row는 toolkit12.x, driver D1, wheel W1, SM90, extension E native sm90, host ABI H1, selected key K native라고 하자. candidate는 toolkit13.x, driver D2, W2, same SM90, E native sm90+PTX compute90, host ABI H2다.

SM90/K native만 보면 candidate도 가능하다. 하지만 H2가 newer PyTorch ABI를 요구하고 production runtime은 H1이면 import에서 실패한다. driver/SM/device inventory를 보기 전에 host leaf가 reject다. higher CUDA target coverage가 무의미하다.

다른 pool SM100에서 host H2는 맞고 W2에 compatible native가 없지만 PTX path가 validated라면 JIT leaf다. cold latency와 driver requirement를 적용한다. W1은 device gate reject일 수 있다. 같은 candidate image도 pool별 leaf가 다르다.

rare key Q가 W2 build gate에서 제외됐다면 common K native success와 별 row다. readiness가 common만 실행하면 Q 첫 request에서 fail/fallback한다. traffic key distribution과 worst important keys를 matrix에 넣는다.

rollback W1은 SM90/H1 pool에는 가능하지만 SM100/H2 pool에는 host/device 둘 다 맞지 않을 수 있다. traffic routing과 framework runtime도 함께 되돌리거나 SM100 pool을 drain해야 한다. rollback bundle은 one wheel가 아니다.

이 연습은 “13>12” 부등식이 왜 쓸모없는지 보여 준다. 각 row는 host ABI, gate, native/PTX/JIT와 production key가 모두 맞아야 한다. 숫자는 predicate inputs 중 일부일 뿐이다.

**최종 incident evidence ladder.**

첫 층은 official facts다. release/compatibility/compiler docs의 exact version, minimum/range, target/host support를 링크한다. 문서에 없는 wheel-specific coverage를 official fact로 쓰지 않는다.

둘째 층은 build/distribution facts다. source commit, normalized version/targets, build logs, wheel/`.so` digest, dependency and code-object manifest다. filename에서 inventory를 추정하지 않는다.

셋째 층은 host runtime facts다. Python/framework versions, resolved DSO paths/symbols, import/op registration이다. loader failure와 success를 기록한다.

넷째 층은 device selection facts다. GPU SM/driver, backend gate, key, native/PTX/JIT/fallback leaf, module/launch logs다. common and failing key를 나란히 둔다.

다섯째 층은 service result다. cold/warm latency, output parity, fallback SLO와 request terminal이다. loader/kernel success가 application correctness를 보장하지 않는다.

first divergence는 층끼리 처음 모순되는 지점이다. build manifest에 target 없음, host symbol unresolved, gate reject, selected key no image, PTX unsupported 또는 JIT cache stale 중 하나로 좁힌다. downstream error 문자열만 원인으로 쓰지 않는다.

**회귀 fixture를 production key로 만든다.**

fixture key는 backend 이름만이 아니다. device SM, dtype, head dimension, causal/window, sequence/context threshold, split count, page/block size, quantization와 optional feature를 포함한다. selector가 실제로 다른 symbol/template를 고르는 최소 좌표를 쓴다.

common key는 startup/readiness용이고 rare key는 사고 재현용이다. long threshold K에서 갈렸다면 K-1/K/K+1을 둔다. batch/sequence1과 maximum representative, tail alignment도 포함한다. key마다 expected native/JIT/fallback/reject leaf를 manifest에 둔다.

cold fixture는 empty process/module/JIT cache에서 실행한다. warm fixture는 same artifact restart/cache reuse와 repeated call을 본다. compile/load time, artifact publish와 selected leaf가 expected인지 확인한다. cold를 제외한 steady benchmark로 JIT storm을 숨기지 않는다.

host matrix는 supported Python/PyTorch/base image/libstdc++ combinations을 제한적으로 명시한다. 지원하지 않는 조합은 explicit reject가 기대다. 모든 가능한 host를 우연히 load시키려 하지 않는다. package metadata와 install resolver도 시험한다.

driver/GPU matrix는 fleet distinct rows와 planned new pool을 포함한다. current minimum driver만 테스트하지 않고 rollout candidate driver도 본다. compat package 사용 row는 resolved user-mode library path를 기록한다.

correctness fixture는 native/JIT/fallback paths의 output/logit/kernel reference parity를 본다. deterministic inputs와 tolerances를 고정한다. path가 다르다고 tolerance를 무제한 넓히지 않는다. unsupported는 조용한 wrong answer보다 reject가 낫다.

failure fixture는 missing DSO/symbol, missing native+PTX, unsupported PTX, JIT compiler/cache permission/timeout, stale cache and fallback disabled를 둔다. 올바른 층의 명시적 error와 cleanup을 검증한다. request가 무한 대기하지 않는다.

**rollback terminal을 object별로 센다.**

service object는 accepted requests, streaming frames와 retries다. artifact rollback 전 new admissions을 막고 inflight output commit 상태를 본다. failed first launch 전 output가 없으면 safe retry 가능성이 높지만 provider/tool side effects와 idempotency를 확인한다.

Python/host object는 imported modules, extension handles와 dependent DSOs다. process 내에서 안전한 unload/reload를 지원하지 않으면 worker drain/restart를 사용한다. old/new symbols를 한 process에서 섞지 않는다.

CUDA object는 contexts, loaded modules/functions, streams/events, graphs와 memory workspaces다. inflight kernels가 끝나기 전에 module/artifact를 폐기하지 않는다. captured graph가 function handle을 보유하면 recapture가 필요하다.

JIT object는 compiler child, temporary build, published cache entries, locks와 failure/backoff state다. rollback generation 이후 candidate compiler가 old namespace에 late publish하지 않게 한다. partial artifacts를 atomic하게 격리한다.

fleet object는 node image, driver, compat/user-mode libraries, GPU pool routing와 scheduler labels다. application container만 되돌려 baseline row가 복원되는지 확인한다. driver/GPU가 바뀌었다면 traffic scope를 조정한다.

telemetry object는 artifact/config generation, selected leaf, errors/fallback와 cold/warm counters다. old worker scrape가 사라지고 new readiness fixture가 통과했는지 본다. aggregate error0만으로 unknown keys가 없다고 단정하지 않는다.

rollback completion은 all important production rows가 expected leaf와 output/SLO를 통과하고 old/candidate pending objects가 0인 것이다. restart 성공 또는 process count 회복만으로 완료하지 않는다.

**공식 사실과 inference를 쓰는 문장 예시.**

공식 사실은 “NVIDIA CUDA Compatibility 문서 version X는 조건 Y에서 minor-version compatibility를 정의한다”처럼 출처와 범위를 쓴다. 우리 application에 바로 성공을 약속한다고 확장하지 않는다.

artifact 사실은 “wheel digest W의 extension E manifest에는 symbol key K용 sm90 native가 있고 PTX 여부는 확인됨/미확정이다”처럼 inspection 범위를 쓴다. filename cu13만으로 target을 만들지 않는다.

inference는 “따라서 driver D, SM90 row는 native candidate이며 host ABI가 통과하면 first-launch될 것으로 예상한다”처럼 전제와 아직 관측하지 않은 leaf를 밝힌다. canary 뒤에는 observed native leaf로 갱신한다.

negative fact도 정확히 쓴다. inspection tool 부재는 “PTX가 없다”가 아니라 “PTX inventory 미확정”이다. source build target omission과 runtime no-image가 함께라면 missing coverage 가설이 강하지만 binary 확인 범위를 명시한다.

release note는 user effect를 붙인다. 어떤 GPU/key가 native/JIT/fallback/reject로 바뀌고 cold/steady latency와 rollback routing이 무엇인지 쓴다. CUDA major 숫자만 강조하지 않는다.

**최종 45분 migration audit.**

첫 10분에는 baseline/candidate coordinates를 수집한다. source/wheel/container, nvcc/toolkit/framework, dependencies, driver/GPU와 target manifests다. 다음 10분에는 host import/DSO/symbol을 닫는다.

세 번째 10분에는 device gate와 production key matrix를 만든다. common/rare key의 selected backend/symbol과 native/PTX/JIT/fallback leaf를 적는다. unknown은 canary/reject로 둔다.

네 번째 10분에는 cold/warm first launch, parity와 latency를 검사한다. JIT cache publish/restart, graph capture와 workspace도 포함한다. 마지막 5분에는 rollback objects와 trigger/readiness를 rehearsal한다.

감사 결과는 one-line pass가 아니라 row table과 evidence links다. 변경되지 않은 rows는 previous attestation을 재사용할 수 있지만 driver/GPU/toolkit/wheel/key predicate 중 하나가 바뀌면 affected rows를 다시 검증한다.

이 절차는 실행 도구를 쓸 수 없을 때도 다음 행동을 구체화한다. official docs와 pinned source/build manifest로 expected tree를 만들고 binary/JIT/runtime leaf는 미검증으로 표시한다. 실제 deployment 전에 필요한 inspection/canary를 release gate에 둔다.

독자가 이 표를 채울 수 있으면 “CUDA 13이 CUDA 12보다 높으니 된다”거나 “driver만 올리면 된다”는 조언을 거른다. 어떤 artifact, symbol, image와 ABI가 어느 경로에서 선택되는지를 기준으로 안전한 build·배포·rollback을 결정한다.

**실제 option과 environment 오진을 교정한다.**

`CUDA_HOME`을 candidate toolkit로 바꾸면 source build/JIT compiler discovery가 달라질 수 있다. 이미 설치된 wheel의 embedded cubin/PTX와 host ABI는 바뀌지 않는다. rebuild가 수행됐는지 artifact digest로 확인한다. environment 값 변화만으로 migration 완료라고 쓰지 않는다.

`PATH`의 nvcc와 framework가 build된 CUDA runtime family가 다를 수 있다. source JIT가 framework headers/libs와 어느 compiler를 결합하는지 project resolver를 읽는다. version warning을 무시하고 compile 성공만 보는 것은 host/device ABI risk를 남긴다.

architecture environment list를 늘리면 emitted targets가 늘 수 있지만 compiler가 해당 target을 지원해야 한다. virtual PTX를 함께 넣는 정책과 binary size/build time trade-off를 본다. unsupported string을 silently drop하는 build가 있는지 manifest를 검증한다.

`LD_LIBRARY_PATH`로 newer CUDA libraries를 앞세우면 intended bundled DSOs를 덮을 수 있다. undefined symbol을 우연히 해결하거나 새 mismatch를 만들 수 있다. resolved paths와 SONAME/version을 저장하고 broad path mutation을 permanent fix로 삼지 않는다.

backend force option은 capability gate/fallback을 우회할 수 있다. unvalidated new SM에서 강제로 op를 선택하면 no-image 또는 wrong behavior가 나타날 수 있다. forced path는 coverage/parity fixture가 통과한 경우에만 recipe로 제공한다.

JIT disable은 cold latency/compile failure를 없애는 대신 AOT coverage 밖 key를 reject/fallback한다. JIT enable은 compiler/headers/cache/permissions와 supply-chain surface를 추가한다. production policy에 맞는 leaf를 명시한다.

compat package 설치는 documented forward-compat scenario에서 user-mode driver components를 제공하는 별 mechanism이다. 모든 older driver/newer toolkit 조합을 해결하는 만능 패치로 사용하지 않는다. supported GPU/OS/package and resolution paths를 official docs로 확인한다.

**supply-chain과 재현성까지 붙인다.**

wheel digest뿐 아니라 build container/base, source submodule pins, compiler packages, CMake flags와 target list를 attestation한다. optional extension이 network/build failure로 빠졌는데 wheel build가 성공할 수 있다면 included/excluded manifest를 release gate로 둔다.

package resolver가 CPU/generic/다른 CUDA variant를 선택할 수 있다. index priority, environment markers와 dependency solver result를 lock한다. pod 안 실제 installed distributions/digests를 startup inventory로 남긴다.

binary inspection output도 artifact로 보존한다. code objects/targets, PTX presence/version, exported host symbols와 dependency paths를 machine-readable manifest로 만들 수 있다. inspection failure는 release unknown으로 처리한다.

JIT produced binary도 attestation 대상이다. source/compiler/flags/SM digest와 output checksum을 cache entry에 붙인다. shared cache에서 untrusted/stale artifact를 load하지 않는다. publish 권한과 cleanup policy를 둔다.

rollback artifact는 미리 보존한다. old wheel/container/dependency lock과 compatible node routing, JIT namespace and fixture results가 있어야 실제 되돌릴 수 있다. package index에서 다시 받아 동일 artifact라고 가정하지 않는다.

**최종 완료 조건.**

CUDA 12.x/13.x 공식 diff는 source link와 범위가 있다. build manifest는 nvcc/toolkit/host compiler, real/virtual targets와 extension inventory를 갖는다. host matrix는 Python/framework/libstdc++/CUDA DSOs를 닫는다.

device matrix는 each SM×production key의 native/PTX/JIT/fallback/reject leaf를 갖는다. common/rare cold/warm launches와 output parity, latency가 검증됐다. unknown은 traffic을 받지 않는다.

incident는 package label이 아니라 selected long key의 missing compatible image에서 first divergence를 찾았다. higher driver/toolkit이 missing artifact를 만들지 않는다는 반증이 있다. fix와 rollback은 exact artifact generation을 사용한다.

service requests, imported modules, CUDA modules/graphs/events, JIT children/cache and telemetry가 terminal됐다. rollback readiness가 common/rare keys를 다시 실행한다. old/new fleet coordinates가 혼용되지 않는다.

마지막 문장은 조건부다. “CUDA 13에서 지원한다”가 아니라 어느 official compatibility 조건, host ABI, GPU SM와 exact production keys가 어떤 leaf로 검증됐는지 쓴다. 미검증 GPU/key는 별 row다.

이 완료 조건을 충족하면 version upgrade는 희망적 숫자 비교가 아니라 source→artifact→loader→driver→request의 재현 가능한 선택 과정이 된다. 장애가 나도 바꿔야 할 owner와 안전한 rollback 범위를 즉시 고를 수 있다.

**마지막 판정 예시.**

SM90 pool은 driver D2, wheel W2와 extension E의 key K/Q 모두 native cubin을 선택했고 host ABI H2, output parity와 cold/warm SLO를 통과했다고 하자. 이 row는 native-validated다. local nvcc 유무는 runtime 판정에 필요하지 않다.

SM100 pool은 common K가 embedded PTX로 JIT되고 rare Q는 device gate가 explicit reject한다고 하자. K는 validated-JIT이며 driver/JIT cache 조건을 붙인다. Q는 unsupported/reject이지 CUDA 13 전체 실패가 아니다. traffic router가 Q를 보내지 않는지 검증한다.

다른 worker는 same W2지만 old PyTorch H1에서 undefined symbol로 import 실패했다. 이 row는 host reject다. GPU SM/driver와 PTX inventory를 바꿔도 import 전에 실패한다. framework-aligned wheel 또는 runtime으로 되돌린다.

세 rows를 “CUDA 13 지원” 하나로 합치면 host failure, JIT cold cost와 rare-key reject가 사라진다. 표에는 wheel/node/request key가 모두 필요하다. production traffic distribution을 적용해 admission과 SLO를 결정한다.

rollback에서는 SM90을 verified W1/H1 row로, SM100은 traffic drain 또는 verified fallback row로 돌린다. JIT namespace와 graphs/modules를 generation별로 폐기한다. old artifact가 D2 driver 조건에서 여전히 공식 범위 안인지 확인한다.

최종 dossier에는 이 row table, official compatibility links, exact digests와 first-launch traces가 있다. 다음 담당자는 version string을 추측하지 않고 어느 predicate가 바뀌었는지 diff할 수 있다.

확정된 사실과 추론도 다시 표시한다. native/JIT leaf는 canary 관측 사실, untested Q performance는 미검증, official driver range는 문서 사실이다. 각각을 섞지 않아야 migration 설명이 과장되지 않는다.

이것이 “높은 버전이면 된다”를 대체하는 운영 문장이다. 더 높은 숫자가 아니라 정확한 host·device·artifact·key 조합과 검증된 선택 경로가 실제 호환성의 재현가능한 판정 단위다.

모든 지원 선언은 이 판정표와 동일한 artifact digest를 인용해야 한다.

## 44.14 장말 소스 노트: toolkit label과 실행 image를 분리한다

### 44.14.1 오류 문구에서 artifact 여정의 첫 좌표를 고른다

`undefined symbol`이면 CUDA 호환성 표보다 package의 host dependency와 loader 경계를 먼저 연다. `no kernel image`이면 build target inventory와 해당 symbol의 cubin coverage를, unsupported PTX이면 PTX version과 driver JIT 계약을 본다. 오류 없이 느린 fallback이면 framework dispatcher와 AOT architecture gate에서 실제 leaf가 바뀐 지점을 먼저 찾는다.

읽기 순서는 source→compile target→package member→loader/dispatcher→driver-selected image다. 아래 공식 문서는 application·minor-version·forward compatibility의 허용 범위를 판정할 때 사용하고, vLLM·SGLang·FlashInfer·llama.cpp 좌표는 그 범위 안에서 실제 배포 artifact와 architecture predicate가 무엇을 만들고 고르는지 확인할 때 사용한다. toolkit 숫자 하나로 두 증거를 합치지 않는다.

### 44.14.2 증상별 고정 source 좌표

- [NVIDIA CUDA C++ Programming Guide 12.9.1 — application compatibility](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#application-compatibility)
- [NVIDIA CUDA Compatibility 13.0.2 — minor-version compatibility](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compatibility/index.html#minor-version-compatibility)
- [NVIDIA CUDA Compatibility 13.0.2 — forward compatibility](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compatibility/index.html#forward-compatibility)
- [vLLM v0.27.1 — CUDA version source and package mapping](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/setup.py#L1008-L1015)
- [SGLang v0.5.18 — AOT CUDA architecture and version gates](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/kernels/aot/CMakeLists.txt#L108-L177)
- [FlashInfer v0.6.17 — explicit AOT architecture requirement](https://github.com/flashinfer-ai/flashinfer/blob/a0a6b019b9b27d49d209f85d028a1ae5a9b347d7/flashinfer/aot.py#L825-L875)
- [llama.cpp v0.2.0 — real·virtual CUDA target inventory](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/ggml-cuda/CMakeLists.txt#L1-L58)

### 44.14.3 kernel artifact의 여정을 한 요청까지 추적한다

CUDA source는 배포 순간 곧바로 GPU에서 실행되는 명령이 아니다. build가 virtual target의 PTX와 real target의 cubin을 만들고, fatbinary와 host object가 이를 운반하며, package가 정확한 member를 배포하고, runtime dispatcher가 요청 shape에 맞는 symbol을 고른 뒤, driver가 compatible native image를 선택하거나 PTX를 JIT해야 비로소 실제 SASS 실행에 도달한다. 따라서 “CUDA 13 wheel”이나 “SM90 지원”은 이 여정 가운데 한 칸의 표지일 뿐 끝까지 닫힌 실행 증명이 아니다.

한 요청 R44를 조사할 때는 source commit과 version source에서 시작해 compile target, artifact digest, host dependency, backend predicate, specialization key, module load, driver image selection과 첫 launch를 순서대로 적는다. `undefined symbol`은 host loader에서, `no kernel image`는 해당 symbol의 device-code coverage에서, unsupported PTX는 driver JIT에서, silent fallback은 dispatcher predicate에서 처음 갈라진다. 오류가 보인 시각과 원인이 생긴 칸을 분리하면 무의미한 driver·nvcc 교체를 피할 수 있다.

CUDA 12.x에서 13.x로 옮기는 일도 같은 ledger의 두 행을 비교하는 작업이다. toolkit 숫자만 바꾸지 말고 compiler path, target inventory, host runtime, driver, compat package, selected backend와 cold/warm artifact path를 고정한다. native cubin이 선택된 결과와 PTX JIT가 선택된 결과를 한 성공으로 합치지 않고, source JIT와 persistent cache가 개입하면 compiler·source·architecture digest까지 generation identity에 넣는다.

마지막으로 지원 여부는 package가 아니라 production symbol/key의 집합에 대해 말해야 한다. 각 key가 native, 검증된 PTX/JIT, 검증된 fallback, 명시적 reject 또는 unknown 가운데 어디에 놓이는지 기록하고, unknown은 조용히 preferred path로 흘려보내지 않는다. 이 표가 build manifest, canary trace와 rollback bundle에서 같은 artifact identity를 가리킬 때 우리는 “어떤 코드가 이 GPU에 도달했는가”를 설명할 수 있다.
