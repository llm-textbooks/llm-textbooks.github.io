# 49장. 파일은 있는데 weight가 없다: shard에서 parameter까지

모델 디렉터리에는 `model-00001-of-00008.safetensors`부터 여덟 번째 shard까지 모두 있다. `model.safetensors.index.json`도 있다. 그런데 loader는 세 번째 shard가 없다고 말한다. 파일명을 눈으로 보면 분명 존재한다. 다시 내려받자 이번에는 시작되지만 `layers.17.mlp.down_proj.weight` shape가 맞지 않는다. 다른 서버에서는 오류 없이 시작했는데 출력이 reference와 전혀 다르다.

세 사건은 모두 “weight loading 실패”로 묶이지만 최초 divergence는 다르다. 첫 사건은 index가 가리키는 filename과 실제 snapshot의 file set이 다를 수 있다. Unicode나 path, stale symlink, 다른 revision의 index가 섞였을 수도 있다. 둘째 사건은 source tensor shape와 TP shard, transpose/packing conversion, destination slice 중 하나가 다르다. 셋째 사건은 같은 이름과 shape를 가진 다른 revision의 byte가 섞였거나 duplicate key가 조용히 뒤의 값으로 덮였을 수 있다.

GGUF에서도 비슷해 보이는 장애가 난다. `model-00002-of-00004.gguf`를 직접 열었더니 “첫 split으로 load하라”는 오류가 난다. 네 파일이 모두 있는데 split count가 다르다고 한다. quantized tensor의 offset을 logical element 수×4bit로 계산했더니 다음 tensor의 시작과 맞지 않는다. GGUF split은 Hugging Face의 `weight_map`과 다른 protocol이고, quant block과 alignment를 모르면 byte range를 계산할 수 없다.

이 장은 parameter 하나의 생애를 끝까지 따른다.

`resolved revision → index/directory → owning shard → header tensor entry → byte range → mapped/read storage → source name → conversion edge → TP·EP slice → dtype/device destination → coverage report`

## 49.1 model load timeline은 revision discovery와 index에서 시작한다

운영자는 model ID와 `main`을 입력해 서버를 시작했다. Loader가 config→index→shard→parameter를 찾는 동안 운영자가 실제로 알고 싶은 것은 “파일이 있는가”가 아니라 “이 model ID가 어느 immutable revision과 file set으로 해석됐는가”다. 이 질문을 먼저 고정해야 뒤의 loader 명사 사슬이 자신의 사고 기록과 연결된다.

### 49.1.1 `main`은 재현 가능한 주소가 아니다

model ID와 `main` branch를 입력하면 hub client는 어느 commit snapshot을 resolve한다. config, tokenizer, index, shards는 같은 resolved commit에 속해야 한다. cache에서는 snapshot directory가 blob symlink를 가리킬 수 있다. 사용자는 평범한 local path를 보지만 그 path가 어느 commit을 가리키는지가 content identity다.

incident 기록에 model ID와 revision 문자열만 적으면 부족하다. 실제 resolved commit hash, index content hash, 각 shard filename·size·content hash를 manifest로 묶는다. `main`은 조사 사이에 움직일 수 있고 tag도 서버에서 다른 commit을 가리키도록 바뀔 가능성을 운영 정책에서 고려해야 한다. 40자리 resolved commit이 증거다.

local directory도 안전하지 않다. 이전 download의 `model.safetensors`, 새 download의 sharded files, 오래된 index가 함께 남을 수 있다. glob으로 모든 `.safetensors`를 loader에 넘기면 같은 tensor가 consolidated와 shard에 중복될 수 있다. index가 있으면 index가 선택한 file set만 사용해야 한다.

configuration revision과 weight revision을 별도로 지정할 수 있는 framework도 있다. 둘이 의도적으로 다를 수 있지만 name/shape contract가 맞아야 한다. model architecture config는 hidden size 4096인데 weight shard는 다른 revision의 5120 shape라면 file parser는 정상이어도 materialization에서 실패한다.

### 49.1.2 Transformers가 checkpoint 후보를 고르는 순서

Transformers current source의 [`_get_resolved_checkpoint_files` 535–650행](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L535-L650)은 local directory인지 remote model인지, explicit filename인지, GGUF file인지, safetensors를 선호/강제하는지 판단한다.

local directory에서 safetensors 사용이 금지되지 않았고 single `model.safetensors`가 있으면 그것을 먼저 찾는다. 없고 index가 있으면 sharded checkpoint로 표시한다. `use_safetensors=True`인데 해당 file이 없으면 `.bin`으로 조용히 fallback하지 않고 오류를 낸다. option의 의미가 preference인지 requirement인지 값을 따라 읽어야 한다.

explicit `transformers_weights` filename은 `.safetensors` 또는 index suffix를 검사하고 base model directory 밖으로 빠져나가지 못하도록 absolute path containment를 확인한다. filename option은 단순 basename override가 아니라 security boundary다. index entry 내부 filename도 별도의 containment/validation이 필요한지 source 경계를 확인한다.

remote path에서는 requested revision과 commit hash, cache directory, local-only, force-download 같은 state가 resolved file을 바꾼다. 같은 process가 이전에 resolve한 commit hash를 후속 shard fetch에 전달해야 index와 shards가 한 snapshot에 고정된다. 각 call이 `main`을 새로 resolve하면 download 도중 branch가 움직이는 race 가능성이 생긴다.

### 49.1.3 index는 tensor 위치가 아니라 owning file을 말한다

Hugging Face index의 핵심은 `weight_map`이다. key는 state-dict tensor name, value는 그 tensor를 담은 shard filename이다. `metadata.total_size`가 있을 수 있지만 loader correctness는 name→file mapping에 의존한다.

다음 fixture를 보자.

```json
{
  "metadata": {"total_size": 24},
  "weight_map": {
    "a.weight": "model-00001-of-00002.safetensors",
    "b.weight": "model-00002-of-00002.safetensors",
    "c.bias": "model-00001-of-00002.safetensors"
  }
}
```

tensor name은 세 개지만 unique shard는 두 개다. `weight_map.values()`의 set을 만들어 file set을 얻는다. dictionary insertion order가 shard load order 계약이라고 가정하지 않는다. 실제 byte offset은 index에 없고 각 safetensors shard header에 있다.

Transformers의 [`get_checkpoint_shard_files` 859–910행](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/utils/hub.py#L859-L910)은 index를 읽고 unique filenames를 정렬하며 `all_checkpoint_keys`와 copied `weight_map`을 metadata에 붙인다. local directory면 subfolder와 결합한 path를 돌려주고 remote면 같은 revision/cache 조건으로 fetch한다.

index가 `a.weight→shard1`이라고 말해도 shard1 header에 실제 key가 있는지 아직 확인하지 않았다. file 존재 set 검사는 packaging incomplete를 빠르게 잡지만 index/header mismatch는 header key set을 대조해야 잡는다. filename이 맞고 content가 다른 revision이면 size/hash manifest가 필요하다.

### 49.1.4 duplicate consolidated file을 제거하는 이유

vLLM current source의 [`filter_duplicate_safetensors_files` 577–608행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/weight_utils.py#L577-L608)은 index가 있으면 weight_map이 참조한 file set을 만들고 glob 결과를 그 set으로 filter한다. referenced file이 glob 결과에 없으면 `FileNotFoundError`를 낸다.

Mistral 계열 repository처럼 sharded files와 consolidated safetensors가 함께 있을 수 있다. 둘을 모두 iterator에 넣으면 같은 parameter name이 두 번 yield된다. model-specific loader가 duplicate를 검사하지 않고 `copy_`하면 마지막 file의 byte가 이긴다. load order에 따라 결과가 달라지는 silent bug다.

따라서 duplicate는 “같은 filename 두 번”만 뜻하지 않는다. 같은 source name이 여러 files에 존재하는지, 여러 source names가 같은 destination slice에 쓰이는지, tied alias가 의도적으로 한 storage를 공유하는지를 분리한다. index-selected set은 첫 종류의 위험을 줄이는 경계다.

**한 번의 model load를 discovery에서 device commit까지 시간순으로 잇는다.**

### 49.1.5 사건의 조건

한 팀이 `acme/MoE-32B` revision `release-2026-08`을 vLLM TP=2, EP=4로 올린다고 하자. repository에는 config와 `model.safetensors.index.json`, 16개 shards가 있다. 각 shard는 약 3 GiB이고 expert weights가 여러 files에 섞여 있다. `safetensors_load_strategy`는 default다.

rank 0과 rank 1은 같은 shared network filesystem cache를 본다. 첫 시도에서 rank 2가 `layers.11.experts.14.w2.weight`를 load하지 않았다고 보고한다. rank 0은 load를 완료했다. directory를 보면 16 files가 모두 있다. 이 사건을 “NFS가 느리다”거나 “EP filter bug”라고 바로 부르지 않는다.

조사 노트의 첫 줄에는 model ID, requested tag가 아니라 resolved commit `R`, config hash `C`, index hash `I`, 16 shard hashes `S1..S16`, framework commit, TP/EP rank mapping을 적는다. 네 ranks가 모두 같은 R/I를 보았는지 확인한다. rank 2 cache symlink가 이전 snapshot `R-1`을 가리키면 loader source를 보기 전에 identity divergence가 닫힌다.

#### 49.1.5.1 index가 file ownership을 얻는다

resolved index에서 target source name을 찾는다. weight_map은 target을 shard 9에 배정한다. unique file set은 16개이고 actual directory와 exact match한다. consolidated file이 하나 더 있지만 filter 뒤 iterator file list에는 없어야 한다.

shard 9의 safetensors header key set을 읽는다. target key가 없다면 index와 shard content가 mixed revision이다. key가 있으면 stored dtype/shape/offset tuple과 shard size를 기록한다. index existence check가 통과했다는 사실은 이 header consistency를 보장하지 않았다.

target offsets가 `[BEGIN,END]`, header length가 N이면 absolute range를 `[8+N+BEGIN,8+N+END)`로 계산한다. END가 data buffer 안에 있고 byte length가 stored dtype×shape와 맞는지 확인한다. 여기까지 맞으면 file packaging과 structural byte contract는 통과다.

#### 49.1.5.2 EP filter가 byte를 읽을지 결정한다

rank 2의 local expert set을 placement plan에서 얻는다. global expert 14가 rank 2 소유인지, replicated expert인지, EPLB mapping으로 다른 local index에 배치됐는지 본다. source name parser가 layer 11과 expert 14를 올바르게 분리하는지 test한다.

expert 14가 rank 2에 필요 없다면 skip은 정상이고 destination에도 해당 independent parameter가 없어야 한다. missing report가 global model parameter set을 expected로 사용해 remote experts까지 요구했다면 validation 집합이 틀린 것이다. local destination inventory를 기준으로 해야 한다.

expert 14가 필요하다면 iterator의 `should_skip_weight` 결과를 본다. prefix가 붙기 전 source name과 parser pattern이 맞는지, shared expert mapping이 skip set에 반영됐는지 확인한다. `get_tensor` 이전 skip이면 target byte range는 page fault/read되지 않았을 수 있다. 이 지점이 IO optimization과 correctness가 만나는 경계다.

#### 49.1.5.3 lazy source의 소유권

normal safe-open path가 shard 9를 열고 header keys를 순회한다. target이 skip되지 않으면 `get_tensor`가 source tensor를 만든 뒤 generator가 model loader에 yield한다. generator는 consumer가 copy를 끝낼 때까지 current file context와 tensor storage를 유지한다.

consumer가 tensor reference를 list에 저장하면 shard 9 mapping/page가 다음 shards와 겹쳐 오래 살아 있을 수 있다. destination에 즉시 copy하고 source reference를 놓으면 lifetime이 짧다. target 하나의 ownership timeline에 `file handle owner`, `mapped storage owner`, `yielded tensor owner`, `destination parameter owner`를 적는다.

prefetch가 effective true였다면 target range가 iterator 전에 page cache에 들어왔을 수 있다. 그러나 EP filter가 get_tensor를 skip해도 prefetch는 file 전체 block을 읽었을 수 있다. logical tensor materialization은 줄었지만 physical storage IO가 같은 상황이다. option 효과를 어느 계층의 byte인지 붙여 말해야 한다.

#### 49.1.5.4 name이 destination slice로 바뀐다

source `layers.11.experts.14.w2.weight`는 destination의 local expert index로 mapping된다. W2 logical shape가 `[hidden,intermediate]`인지 checkpoint convention의 transpose인지 확인한다. quantized W2면 qweight/scales/ZP 각각의 source entry와 packed logical shape를 연결한다.

TP=2가 W2 output 또는 input 어느 axis를 나누는지 model parallel layer가 정한다. source full logical shape에서 rank 2 process의 TP rank가 가져갈 interval을 계산한다. EP global→local mapping 뒤 destination tensor는 `[local_expert, packed_K, local_N]` 같은 physical shape일 수 있다.

destination coverage edge를 기록한다. source name, source stored range, logical transform, TP interval, local expert index, destination slice다. loader가 returned loaded-name set에 destination name을 넣었다고 slice가 전부 채워졌다는 뜻은 아니다. qweight와 scale/ZP가 함께 준비돼야 W2 parameter group이 complete하다.

#### 49.1.5.5 첫 시도의 실제 divergence

조사 결과 rank 2의 expert set에는 14가 포함됐다. iterator도 skip하지 않았다. shard/header와 source tensor는 정상이다. 그런데 mapping table은 expert 14를 local index 2로 보내고 destination coverage validator는 stale placement plan을 사용해 local index 1을 expected했다.

source byte는 올바른 destination에 들어갔지만 validator가 다른 destination name을 요구해 missing을 보고한 것이다. 이 경우 “strict check를 끈다”가 해결이 아니다. placement plan의 single source of truth를 정하고 IO filter, mapping, expected coverage가 같은 plan revision을 사용하게 한다.

반대 사례도 fixture로 만든다. mapping이 stale이고 validator도 같은 stale plan이면 둘이 합의해 load 성공으로 보이지만 wrong expert weight가 배치될 수 있다. router/expert execution mapping이라는 제3 consumer와 비교해야 한다. validation은 내부 두 모듈의 일치가 아니라 model semantic mapping과 일치해야 한다.

#### 49.1.5.6 실패한 model을 publish하지 않는다

rank 2가 validation error를 내면 다른 ranks가 완료했어도 model group을 serving registry에 올리지 않는다. collective coordination으로 모든 ranks load success를 확인한 뒤 atomic publish한다. 일부 ranks가 requests를 받기 시작하면 distributed execution이 hang하거나 wrong output을 낼 수 있다.

failed instance의 file handles, mappings, thread futures, host buffers, destination GPU allocations을 해제한다. rank 0의 completed model을 그대로 두고 rank 2만 reload하면 placement plan generation과 graph/buffer addresses가 달라질 수 있다. system이 partial-rank retry를 지원한다면 명시적 generation protocol이 필요하다.

cache snapshot 자체가 corrupt하지 않으므로 shards를 다시 download할 이유는 없다. first divergence가 mapping/coverage plan이었다는 증거가 불필요한 48 GiB network read를 막는다. “load 실패=redownload” 운영 절차가 좋지 않은 이유다.

#### 49.1.5.7 같은 model을 SGLang으로 읽는다면

SGLang에서도 index set과 header는 같은 format contract를 따른다. 그러나 buffered multithread option, name mapping implementation, expert loader가 다르다. vLLM에서 발견한 stale plan bug를 SGLang에 있다고 승계하지 않는다.

동일 incident fixture를 source name→local expert set→destination edge 관점으로 재사용한다. SGLang의 actual iterator가 whole-read인지 safe-open인지, multithread queue가 몇 shards를 보유하는지 기록한다. correctness tuple은 같아도 ownership timeline과 peak가 다르다.

SGLang load가 성공하면 checkpoint byte가 옳다는 교차 증거는 되지만 vLLM mapping의 expected destination이 옳다는 증거는 아니다. 두 frameworks의 model implementation이 expert order를 다르게 표현할 수 있다. 최종 reference output과 mapping semantic을 연결한다.

#### 49.1.5.8 GGUF로 변환한 뒤에는 무엇이 바뀌는가

같은 logical model을 GGUF splits로 변환했다고 하자. 이제 HF weight_map 대신 first GGUF의 split count/no와 각 file tensor directory가 ownership을 정한다. source tensor names도 GGUF convention으로 변할 수 있고 quant type/block layout이 conversion 때 고정된다.

incident manifest에는 conversion input snapshot R과 converter commit/options, output GGUF files hashes를 묶는다. GGUF가 self-contained metadata를 갖는다고 conversion provenance가 자동 포함되는 것은 아니다. metadata key에 source revision이 있으면 검증하고 없으면 외부 manifest를 둔다.

expert 14 W2 owning split을 combined directory inventory에서 찾는다. 그 split의 data base와 tensor relative offset, `ggml_nbytes`, alignment-padded interval을 계산한다. HF shard 9 offset을 재사용할 수 없다. quant block type이 바뀌었으므로 stored shape/byte reference도 새로 만든다.

llama.cpp model graph가 global expert ID와 tensor name을 연결하고 backend buffer를 배치한다. EP=4 server와 같은 distribution을 한다고 가정하지 않는다. single-process CPU/GPU offload model이면 all experts가 한 process의 buffers에 있을 수 있다. 같은 logical model도 serving topology와 destination inventory가 다르다.

#### 49.1.5.9 이 사건이 가르치는 조사 순서

이 사건은 file→tensor→parameter를 단방향 복사로 보면 놓치는 것을 보여 준다. source identity와 destination expectation이 각자 다른 plan을 소유할 수 있다. index, header, iterator, name mapper, placement plan, coverage validator, execution consumer가 같은 generation의 의미를 공유해야 한다.

조사자는 모든 것을 한 번에 dump하지 않는다. 먼저 snapshot/index/file set을 닫고, target header/byte range를 닫고, IO filter 결정을 닫고, mapping edge와 destination coverage를 닫는다. 각 단계가 통과할 때 다음으로 이동한다. 이 순서는 remote download, local mmap, eager bytes 어느 path에서도 유지된다.

memory 문제도 같은 timeline에서 푼다. 어느 file들이 in-flight인지, target tensor가 yield된 뒤 얼마나 오래 살아 있는지, destination copy와 conversion temporary가 언제 겹치는지 본다. total checkpoint size 한 숫자보다 객체 lifetime interval이 peak를 설명한다.

마지막으로 failure를 재현 가능한 작은 fixture로 줄인다. expert 4개, EP=2, 두 shards, source expert names와 swapped placement generation을 만든다. 필요한 expert를 skip하는 경우, 올바르게 load하지만 validator만 stale인 경우, mapper와 validator 모두 stale인 silent case를 각각 test한다. 운영 incident의 의미를 unit-level invariant로 바꾼다.

이 축소 fixture를 만들 때는 정상 결과만 assert하지 않는다. 첫 번째 실행에서는 두 shard의 open 순서를 고정하고, 두 번째 실행에서는 순서를 뒤집으며, 세 번째 실행에서는 buffered reader가 완료한 순서대로 tensor를 내보내게 한다. 세 실행의 destination bytes와 coverage edge 집합은 같아야 한다. 결과가 file 순서에 따라 달라진다면 duplicate source, last-wins assignment, mutable mapping table 가운데 하나가 숨어 있다. concurrency를 끄면 문제가 사라지는 현상도 곧바로 thread race를 증명하지 않는다. 순차 실행이 우연히 잘못된 overwrite 순서를 안정화했을 수 있으므로 source-to-destination edge와 assignment count를 함께 비교한다.

fixture의 관찰 기록은 단순한 debug log보다 구조화된 load ledger가 낫다. ledger 한 행에는 `attempt_id`, resolved revision, rank, source file hash, source tensor name, stored dtype와 shape, byte interval, filter decision, transform ID, destination name, destination slice, materialized dtype와 device, copy 완료 여부가 들어간다. 파일 전체를 반복해서 dump하지 않고 실패한 tensor와 인접한 몇 행만 남겨도 원인 사슬을 재구성할 수 있다. 보안상 weight bytes를 기록하지 않아도 hash와 interval로 동일 source를 식별할 수 있다. 다만 tensor checksum은 dtype 변환 전인지 후인지 명시한다.

FP16 source를 BF16 destination으로 바꾼 뒤의 checksum을 원본 checksum과 비교하면 당연히 다르기 때문이다.

ledger는 성능 분석에도 그대로 쓰인다. 각 행에 open 시작, header parse 완료, source materialization, host-to-device copy 시작과 종료, source release 시각을 붙이면 tensor마다 대기와 소유 기간이 나온다. 여기서 shard 9 open은 빠른데 첫 tensor yield가 늦다면 header 문제가 아니라 prefetch 또는 storage read가 의심된다. source yield는 빠른데 copy가 늦다면 destination allocation, dtype conversion, PCIe 전송, 이전 stream 작업을 분리한다. load 전체 소요 시간 하나만 보면 이 경계들이 모두 사라진다. “loader가 느리다”는 문장은 어느 상태 전이가 지연됐는지 쓰기 전에는 진단 문장이 아니다.

메모리 peak도 ledger의 interval 합으로 검산한다. 시각 t에서 살아 있는 객체를 mapped file pages, eager host bytes, pinned staging, converted temporary, destination GPU storage로 나누고 각 크기를 합한다. 운영 metric의 RSS에는 page cache와 allocator accounting이 다르게 보일 수 있으므로 이 합이 RSS와 정확히 같을 필요는 없다. 중요한 것은 option 변경 전후에 어느 항이 늘었는지다. worker를 1에서 8로 늘린 뒤 eager host bytes가 여덟 shard만큼 겹치고 처리량은 거의 그대로라면 bottleneck은 open parallelism이 아니다. 반대로 mapped pages만 늘고 major fault latency가 줄었다면 storage queue가 병렬 요청을 흡수했을 가능성이 있다.

같은 “메모리가 늘었다”라도 원인과 조치가 다르다.

option 실험은 한 번에 하나씩 하고 상태 전이를 적는다. eager 전략을 켜면 source representation이 file-backed view에서 owned host tensor로 바뀌고, 그 결과 handle lifetime은 짧아질 수 있지만 host allocation과 copy가 추가된다. prefetch를 켜면 향후 접근할 byte를 storage 또는 page cache에 먼저 요구하지만 destination allocation 시점은 자동으로 바뀌지 않는다. worker 수를 늘리면 in-flight shard 상한과 completion 순서가 변한다. load format을 GGUF로 바꾸면 같은 reader의 다른 mode가 아니라 metadata, name convention, quant block, destination backend까지 다른 계약을 선택한다.

따라서 실험표에는 option 이름 옆에 바뀐 객체, queue, byte path, expected metric, correctness invariant를 한 줄씩 쓴다.

예를 들어 `workers=8`의 기대 효과를 “더 빠름”이라고 적지 않는다. shard open과 read가 독립적이고 storage가 queue depth를 받아들일 때 wall time이 줄 수 있다고 쓴다. 동시에 최대 여덟 source payload와 futures가 살아 peak host memory가 증가할 수 있고, yield ordering이 달라져 duplicate bug가 드러날 수 있다고 쓴다. 관찰할 metric은 shard read latency distribution, bytes in flight, queue wait, host allocation peak다. 불변식은 source inventory, destination coverage, final parameter checksum이 worker 수와 무관하다는 것이다. 이 정도로 적어야 옵션이 magic knob가 아니라 검증 가능한 가설이 된다.

`mmap`도 같은 방식으로 읽는다. mmap 자체는 tensor를 GPU에 보내지 않고, file byte range를 process virtual address에 연결한다. 실제 page가 언제 resident가 되는지는 접근 패턴과 운영체제에 달렸다. 그래서 open 시간이 짧다는 관찰만으로 load가 빨라졌다고 결론 내릴 수 없다. 첫 inference까지 포함한 major page fault가 뒤로 이동했을 수 있다. warm page cache benchmark와 cold cache benchmark를 나누고, mapped virtual size와 resident pages를 구분하며, source tensor가 destination에 복사된 뒤 mapping이 해제되는지 확인한다.

GPU destination이 완전히 materialize되는 loader라면 mmap의 주된 이점은 host-side 중간 복사와 접근 선택성에 있고, model weight가 file에서 GPU 연산으로 곧바로 흐른다는 뜻은 아니다.

direct IO라는 표현도 엄격히 제한한다. GGUF reader가 aligned byte range를 backend buffer에 읽는 경로와, 일반 buffered file IO 뒤 memcpy하는 경로는 다를 수 있다.

그러나 alignment 조건을 맞췄다는 사실만으로 storage에서 GPU memory까지 복사 없이 이동한다고 부르면 안 된다. filesystem, kernel buffer, registered host memory, DMA 대상, backend upload 단계 가운데 실제로 생략된 것을 증명해야 한다. quantized tensor의 requested interval은 block boundary와 file alignment를 동시에 만족해야 하며, logical element 구간만 잘라 읽으면 앞 block의 scale과 quant payload를 잃을 수 있다. direct-read optimization은 format-aware interval planner와 함께 검증한다.

분산 환경에서는 모든 rank의 ledger를 attempt ID와 barrier generation으로 묶는다. rank 0이 snapshot R을 확인한 뒤 rank 3이 cache miss 때문에 R의 shard를 새로 받는 동안 repository pointer가 움직여도, rank 3은 이미 확정된 content 주소만 사용해야 한다. 각 rank가 독립적으로 `main`을 resolve하면 같은 load attempt 안에서 config와 weights가 갈릴 수 있다. shared filesystem을 쓴다고 identity가 자동 통일되는 것도 아니다. symlink resolution 시각, local cache completeness, stale directory entry가 다를 수 있다. coordinator는 resolved manifest를 배포하고 ranks는 manifest의 file size와 hash를 검증한 뒤 read phase에 들어간다.

검증 barrier는 “파일이 존재한다”에서 끝나지 않는다. 첫 barrier는 manifest completeness, 두 번째는 header와 directory structural validity, 세 번째는 local destination coverage, 네 번째는 collective model readiness를 닫는다. 구현이 이 네 단계를 실제 barrier 네 개로 만들 필요는 없지만 실패 메시지는 어느 계약이 닫히지 않았는지 보여 줘야 한다. rank 2 header failure를 마지막 collective timeout으로만 보면 운영자는 NCCL 문제를 쫓게 된다. 최초 local error를 보존하고 다른 ranks의 취소 원인을 derivative failure로 표시하면 storage corruption과 distributed coordination failure를 구분할 수 있다.

부분 다운로드를 더 현실적으로 생각해 보자. 파일명과 expected size는 맞지만 sparse range 일부가 아직 fetch되지 않은 remote filesystem도 있다. header와 앞 tensor들은 정상이고 뒤 tensor에서만 short read 또는 checksum mismatch가 난다. 단순 `stat` 검사는 통과한다. loader는 read가 요청한 정확한 byte count를 반환했는지 검사하고, remote cache layer는 materialized range 상태를 노출해야 한다. 재시도할 때도 전체 snapshot을 지우기보다 손상된 content object나 range를 식별한다. 반면 cryptographic hash가 file 전체 불일치를 보이면 부분 범위를 추정하지 말고 해당 immutable object를 교체한다. 복구 범위는 증거의 해상도보다 좁아서는 안 된다.

mixed revision은 shape mismatch로 친절하게 실패할 때보다 shape가 우연히 같을 때 더 위험하다. tokenizer 또는 config는 R2인데 weights는 R1이고 두 revision의 hidden size가 같다면 byte-level validation은 모두 통과할 수 있다. rope scaling, expert order, vocabulary semantic처럼 shape 밖의 의미가 달라져 출력만 틀릴 수 있다.

그러므로 content manifest에는 config, tokenizer assets, chat template, index, shards를 하나의 revision closure로 묶는다. loader 장만으로 tokenizer semantic을 검증할 수는 없지만 serving artifact assembler가 closure를 보장하고 load report에 그 identity를 남겨야 한다. “모델이 떴다”와 “의도한 모델이 떴다” 사이를 revision closure가 메운다.

name mapping에서도 문자열 결과만 snapshot test하면 부족하다. transform을 단계별 typed operation으로 표현하면 더 강한 검증이 가능하다. prefix 제거, layer index 해석, expert global-to-local 변환, projection alias 변환, packed component 선택, transpose, TP slice를 순서대로 기록한다. 각 단계는 입력 domain과 출력 domain, shape 효과를 가진다. 두 규칙이 같은 source key에 동시에 match하면 priority로 조용히 하나를 택하기보다 ambiguity를 실패로 만든다. 새 architecture 지원에서 regex 한 줄이 기존 model의 key까지 잡아먹는 회귀를 막는 방법이다.

packed quantization은 coverage 단위를 특히 조심한다. logical weight 하나가 qweight, scale, zero point, group index 여러 destination buffers를 채울 수 있고, 반대로 checkpoint의 fused tensor 하나가 여러 logical projections으로 갈라질 수 있다. source key 존재 여부와 destination parameter name 존재 여부를 일대일 비교할 수 없다.

coverage graph에서 source byte interval이 어떤 semantic component를 제공했고, 각 destination slice가 필요한 component를 모두 받았는지 확인한다. scale만 누락됐는데 qweight 이름이 loaded set에 있다는 이유로 성공 처리하면 kernel은 초기화되지 않은 scale을 읽는다. shape가 맞아도 값이 의미를 잃는 전형적인 silent failure다.

dtype 변환은 별도의 transform edge로 둔다. BF16 source를 FP16 destination에 넣는 것이 허용되는지, FP32 scale을 그대로 유지해야 하는지, quant metadata integer width를 바꿀 수 있는지는 parameter 종류마다 다르다. 전역 `dtype` option 하나가 모든 stored tensor를 같은 방식으로 cast한다고 가정하지 않는다. cast 전후 element count, finite-value policy, overflow 가능성, destination storage dtype를 기록한다. device 이동과 cast가 fused된 경로에서는 어느 단계에서 검증할 수 있는지도 정한다. CPU reference tensor를 오래 보관해 정확도를 검산하면 peak가 늘므로 작은 sentinel slice 또는 deterministic checksum fixture를 사용한다.

정상 종료의 정의도 구체적이어야 한다. 모든 expected destination slice가 정확히 한 semantic source로 채워졌고, 허용된 tied alias와 intentional padding만 예외이며, no pending future, no dangling source handle, no failed rank가 남아야 한다. model registry publish 전에 architecture-level smoke assertion을 수행할 수도 있다. embedding과 한 attention projection, 한 expert, output head처럼 대표 parameter의 shape, dtype, device, checksum identity를 manifest와 비교한다. 이것은 full inference 실행이 아니라 loader가 선언한 결과를 표본으로 다시 확인하는 inexpensive guard다.

다만 표본 검사가 full coverage proof를 대신해서는 안 된다.

운영자가 실제 장애에서 사용할 순서도 이 정의에서 자연스럽게 나온다. 먼저 모든 rank의 resolved identity와 manifest closure를 모은다. 다음으로 owning file과 header directory에서 source tensor의 구조를 확인한다. 그다음 filter가 source를 읽기로 했는지, source lifetime이 copy까지 유지됐는지 본다. 이어 typed mapping edge와 destination slice assignment count를 확인한다. 마지막에 dtype/device materialization과 collective publish를 본다. 앞 단계의 증거 없이 뒤쪽 성능 옵션을 바꾸지 않는다. 이 순서를 따르면 partial file에 worker 수를 조절하거나 stale mapping에 redownload를 반복하는 식의 비용 큰 오진을 피할 수 있다.

이 장의 예시 model에서 최종 수정은 placement plan generation을 immutable object로 만들고, iterator filter, name mapper, coverage validator, execution router가 같은 object ID를 받게 하는 것이다. load report에는 그 ID를 남긴다. 회귀 test는 plan object 하나만 바꾸어 네 consumer가 함께 변하는지 검사한다. rank별 local expert inventory와 destination edge를 canonical ordering으로 직렬화해 비교하면 nondeterministic iteration 차이도 찾을 수 있다. 이 설계는 특정 framework의 함수 이름보다 오래 간다. 핵심은 source selection, destination interpretation, execution interpretation이 한 의미 계약을 공유하도록 만드는 데 있다.

## 49.2 mmap·slice·read가 host byte lifetime을 정한다

### 49.2.1 `safe_open`이 하는 일

`safe_open(file, framework="pt")`은 header를 parse하고 key iterator, metadata, `get_slice`, `get_tensor`를 제공한다. mmap backend는 file byte를 virtual address space에 mapping하여 필요한 page가 접근될 때 page cache를 통해 들어올 수 있다. Python `bytes`로 file 전체를 먼저 복제하는 eager path와 다르다.

“zero-copy”라는 홍보 문구를 CPU→GPU까지 복사가 없다는 뜻으로 쓰면 안 된다. file-backed CPU mapping에서 GPU parameter storage로 materialize하면 device transfer가 필요하다. dtype cast, transpose, TP slice, quant repack이 있으면 destination/temporary allocation도 생긴다. zero-copy 가능 범위는 file→CPU view의 일부다.

mapping은 page cache를 사용하므로 process RSS, system page cache, virtual memory 수치가 다르게 보일 수 있다. file pages가 resident하면 physical RAM을 쓰지만 anonymous tensor copy와 accounting이 다르다. “mmap은 RAM을 쓰지 않는다”고 쓰지 않는다.

### 49.2.2 Transformers 단일 file path

Transformers [`load_state_dict` 319–366행](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L319-L366)은 safetensors file에서 `disable_mmap`과 map location을 해석한다. mmap을 끄고 meta가 아니면 `open(...).read()`로 전체 bytes를 읽고 safetensors bytes loader로 state dict를 만든다.

기본 path는 `safe_open`을 열고 keys를 순회한다. `map_location="meta"`면 `get_slice`로 dtype/shape를 얻어 meta empty tensor만 만든다. 실제 weight bytes를 materialize하지 않는다. meta model initialization과 later destination placement를 설계할 수 있는 이유다.

meta가 아니면 `get_tensor(k).to(map_location)`을 한다. CPU default라면 file view/tensor storage lifetime이 state dict에 남을 수 있고 device로 옮기면 device allocation과 source pages가 겹친다. map location과 backend, safetensors binding version에 따라 copy/view semantics를 확인한다.

### 49.2.3 lazy slices와 handle lifetime

Transformers의 sharded loading path [`modeling_utils.py` 4455–4494행](https://github.com/huggingface/transformers/blob/550d7b3834670483a4df436541272c055dc364bf/src/transformers/modeling_utils.py#L4455-L4494)은 mmap/pread handle을 열어 `get_slice(k)`를 merged dictionary에 넣되 즉시 tensor를 만들지 않는다. conversion/load가 slice를 destination에 넣은 뒤 모든 open pointer를 닫는다.

이 design은 header/name graph를 먼저 만들고 parameter별 필요한 byte를 materialize할 수 있다. device map과 TP plan, conversion mapping이 source slice에서 필요한 부분만 가져갈 가능성을 준다. 그러나 handle을 destination load 전에 닫으면 slice가 invalid할 수 있다. file pointers set의 lifetime이 conversion call을 둘러싸는 이유다.

`disable_mmap` 또는 특정 FUSE mount에서는 shard bytes를 읽어 `_safe_load_bytes` dictionary로 update한다. 이 경우 shard state tensors가 merged dictionary에 쌓이면 여러 shard storage가 동시에 살아 peak가 커질 수 있다. code가 shard마다 destination으로 즉시 copy하고 dictionary를 버리는지, 전체 merge 후 load하는지 구분한다.

### 49.2.4 memory peak 손계산

5 GiB shard 네 개, destination BF16 model 20 GiB를 가정하자. 이상적인 sequential shard load가 shard 하나를 CPU에 materialize하고 곧바로 GPU destination에 copy한 뒤 해제하면 anonymous host peak 후보는 약 5 GiB+conversion temporary이고 device는 점차 20 GiB까지 증가한다.

네 shards를 모두 Python bytes/state dict로 merge한 뒤 device에 넣으면 host source 약 20 GiB와 device destination 20 GiB가 겹칠 수 있다. bytes object와 decoded tensor가 separate copy라면 shard 단위 추가 복제까지 생길 수 있다. 실제 sharing 여부를 allocator/storage pointer로 확인해야 한다.

mmap lazy path는 20 GiB virtual mapping을 가질 수 있지만 접근된 pages만 resident한다. model 전체를 device로 copy하면 결국 file pages가 page cache에 들어올 수 있어 system RAM pressure는 생긴다. anonymous 20 GiB state dict를 반드시 만들 필요는 없다는 차이다.

dtype cast가 FP32 source 40 GiB에서 BF16 destination 20 GiB라면 cast 순간 source pages/storage와 destination이 겹친다. device cast인지 CPU cast인지에 따라 host/device peak가 달라진다. quant repack은 original qweight와 repacked parameter, scale permutation temporary가 겹칠 수 있다.

## 49.3 index가 shard iterator와 owning file을 정한다

### 49.3.1 vLLM의 format 선택

vLLM current source의 [`DefaultModelLoader._get_weights_iterator` 244–319행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/default_loader.py#L244-L319)은 model path, subfolder, revision에서 file list와 safetensors 여부를 준비한다. `load_format`과 extra config가 실제 iterator를 고른다.

`fastsafetensors`, `instanttensor`, normal safetensors, multithread safetensors, npcache, PyTorch iterator가 분기된다. source option의 label만 보고 `safe_open` path라고 단정하지 않는다. effective load format과 installed optional package, extra config를 기록한다.

iterator가 yield하는 단위는 `(source_name,tensor)`다. `Source.prefix`가 있으면 name 앞에 붙는다. primary model 외에 secondary source나 prefix-mounted module이 있을 수 있다. prefix를 file tensor name의 일부라고 오해하면 unexpected key를 만든다.

state dict 전체를 만든 뒤 `load_state_dict`하는 대신 generator를 model-specific `load_weights`에 전달하면 tensor 하나 또는 shard 단위로 destination에 copy하고 source lifetime을 짧게 할 수 있다. 그러나 model loader가 generator item을 모아두거나 conversion을 위해 여러 source를 기다리면 peak가 다시 늘 수 있다.

### 49.3.2 normal, eager, prefetch

vLLM [`safetensors_weights_iterator` 838–972행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/weight_utils.py#L838-L972)은 file을 natural sort하고 filesystem type, checkpoint total bytes, available RAM을 계산한다.

normal path는 shard마다 `safe_open`, key마다 `get_tensor`, yield다. consumer가 tensor를 destination에 넣고 다음 item으로 진행한다면 source tensor lifetime을 제한할 수 있다. generator가 pause된 동안 current file handle과 yielded tensor가 살아 있음을 기억한다.

eager strategy는 `open(st_file,"rb")`와 `load(f.read())`로 shard 전체를 bytes에서 decode한다. 한 shard 안의 key를 yield하는 동안 state dict가 살아 있으므로 shard 전체 storage가 유지될 수 있다. many-small-tensor overhead나 network FS behavior에 이득 가능성이 있어도 peak host RAM은 shard 크기에 민감하다.

prefetch는 file 전체를 application tensor dictionary로 만드는 것과 다르다. block reads로 page cache를 미리 데워 이후 mmap fault를 줄이려 한다. current code는 NFS/NFS4/Lustre이고 total checkpoint가 available RAM의 90% threshold 안이면 auto-prefetch를 선택할 수 있다. local FS에서는 default auto-prefetch를 하지 않는다.

force prefetch가 RAM threshold를 넘으면 warning을 낸다. page cache도 다른 process와 system을 압박하고 useful pages를 evict할 수 있다. “memory copy가 없으니 peak와 무관”이라고 쓰지 않는다. option→should_prefetch effective state→prefetch call을 연결한다.

### 49.3.3 EP expert early skip

MoE expert parallel에서 각 rank는 local experts weight만 필요할 수 있다. iterator는 `local_expert_ids`를 받으면 `get_tensor` 전에 source name을 검사해 remote expert weight를 skip한다. header key는 순회하지만 tensor data page/read와 destination allocation을 줄일 수 있다.

예를 들어 E=64, EP=8이고 expert weight byte가 균등하면 rank당 8 experts가 필요해 expert payload의 ideal 1/8만 materialize할 수 있다. shared layers와 router, duplicated experts, metadata는 그대로다. file sharding이 expert별이 아니어도 key-level lazy read가 data-range IO를 줄일 수 있지만 storage/page-cache prefetch가 file 전체를 먼저 읽으면 physical IO 절감이 약해질 수 있다.

name parser가 `model.layers.3.mlp.experts.17.w1.weight`의 expert 17을 정확히 찾고 expert remapping/replication을 반영해야 한다. prefix나 model-specific naming이 달라 skip logic이 필요한 local expert를 버리면 missing destination이 된다. skip set과 destination expert map을 같은 source에서 만든다.

### 49.3.4 TorchAO cross-shard state

serialized tensor subclass는 한 logical parameter가 여러 flat component와 metadata로 표현될 수 있다. current vLLM torchao path는 shard 하나의 state dict에 이전 shard의 leftover를 합치고 `unflatten_tensor_state_dict`를 호출한다. 필요한 components가 아직 없으면 leftover로 다음 iteration에 넘긴다.

따라서 “shard 하나를 load하면 모든 tensor를 즉시 yield한다”는 invariant가 깨진다. logical object boundary가 file shard를 가로지를 수 있다. 마지막 shard 뒤 leftover가 비어 있는지, duplicate component가 없는지, metadata가 같은 subclass를 가리키는지 검사해야 한다.

cross-shard leftover는 peak와 error timing도 바꾼다. component가 오래 살아 다음 shard storage와 겹친다. missing component error가 해당 file을 읽을 때가 아니라 전체 iteration 끝에서 나타날 수 있다. format-specific reconstruction을 generic safetensors behavior로 설명하지 않는다.

### 49.3.5 loaded-name tracking

[`DefaultModelLoader.load_weights` 414–450행](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/model_loader/default_loader.py#L414-L450)은 model-specific `model.load_weights(iterator)`를 호출하고 returned loaded-name set을 이용할 수 있다.

non-quantized model이고 tracking을 제공하면 strict check를 기본 활성화할 수 있다. quantized model은 checkpoint names와 destination parameters의 mapping이 복잡해 default가 다를 수 있다. strict가 꺼져 있다는 사실을 “모두 성공”으로 해석하지 않는다. quant loader는 자체 coverage proof가 필요하다.

tracking은 source names set이 아니라 destination parameters coverage여야 유용하다. Q/K/V 세 source가 packed `qkv_proj.weight`의 서로 다른 shard slice를 채우고 gate/up 두 source가 `gate_up_proj`를 채운다. destination name 하나를 loaded로 표시하기 전에 모든 slice coverage가 완료됐는지 확인해야 한다.

## 49.4 shard read concurrency와 peak를 함께 제한한다

### 49.4.1 missing shard를 읽기 전에 막는다

SGLang current source의 [`check_safetensors_index_files` 436–471행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/weight_utils.py#L436-L471)은 snapshot directory의 index files를 찾고 각 `weight_map.values()` required set이 실제 directory에 있는지 검사한다.

이 fail-fast는 load 중반까지 GPU parameter 일부를 채운 뒤 file open error가 나는 비용을 줄인다. 하지만 file 존재와 content integrity는 다르다. truncated shard, wrong header, wrong revision content는 다음 validation이 잡아야 한다.

SGLang의 duplicate filter도 index-selected file set을 사용하고 consolidated file을 제외한다. packaging bug로 `mtp.safetensors`가 존재하지만 index에 없을 때 auto-add하는 model-specific 예외가 있다. 일반적인 “unindexed file도 모두 추가” rule로 확장하지 않는다. exception에는 model condition과 경고가 필요하다.

### 49.4.2 sequential safe-open과 eager

[`safetensors_weights_iterator` 1091–1127행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/weight_utils.py#L1091-L1127)은 `load_format="safetensors"`이면 file 전체 bytes를 읽어 `safetensors.torch.load`하고, 아니면 `safe_open(device="cpu")`으로 key를 순회한다.

option 이름이 framework마다 다르다. vLLM의 `safetensors_load_strategy="eager"`와 SGLang의 load-format branch가 비슷한 whole-read behavior를 가질 수 있지만 default, config surface, logging이 다르다. 동일 옵션이라고 쓰지 않는다.

whole-read path에서 `f.read()` bytes와 returned tensors가 storage를 공유하는지 library implementation을 확인한다. peak 모델은 보수적으로 shard bytes와 decoded tensor, destination의 overlap 후보를 둔다. 측정으로 sharing을 확인하기 전 exact 2배라고 단정하지 않는다.

### 49.4.3 unbounded submit을 피하는 buffered iterator

multithread loader가 모든 shard read task를 한꺼번에 submit하고 result state dict를 쌓으면 workers 수보다 많은 completed shards가 memory에 대기할 수 있다. thread pool workers가 4라고 peak가 4 shards로 제한된다는 보장은 없다. producer/consumer queue의 bound를 봐야 한다.

SGLang [`buffered_multi_thread_safetensors_weights_iterator` 1232–1278행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/weight_utils.py#L1232-L1278)은 at most `max_workers+1` shard files가 in-flight라고 설명하고 CPU peak를 대략 `(max_workers+2)×shard_file_size`로 모델링한다.

workers=4, uniform 5 GiB shards라면 source comment의 보수적 peak 모델은 30 GiB다. final model destination이나 conversion temporary는 별도다. shard sizes가 불균등하면 largest in-flight shards 합으로 바꿔야 한다. 평균 shard size를 곱하면 큰 shard 여러 개가 동시에 있을 때 과소평가한다.

consumer가 느리면 completed results가 bounded slot을 차지해 producer가 막혀야 한다. exception이 나면 pending futures를 취소하고 open/read가 종료되는지 본다. corrupt shard 하나 뒤 다른 workers가 수십 GiB를 계속 읽는다면 fail-fast의 운영 효과가 사라진다.

### 49.4.4 thread 수의 손익

network filesystem 또는 object-backed mount에서는 여러 reads가 latency를 겹치고 aggregate bandwidth를 높일 수 있다. local NVMe가 이미 sequential throughput을 내거나 page cache hit라면 threads가 seek/metadata contention과 RAM peak만 늘릴 수 있다. Python decode가 GIL을 얼마나 놓는지도 implementation에 달렸다.

thread option의 설명에는 filesystem, shard count/size, queue bound, total RAM, destination copy rate를 붙인다. load time 하나만 비교하지 않고 peak host memory, bytes read, page faults, failed-load cleanup을 본다. 이 장은 결과 숫자를 제시하지 않고 observation contract만 만든다.

## 49.5 shard tensor를 destination device parameter로 materialize한다

### 49.5.1 source key와 destination parameter

checkpoint source name은 serialization architecture의 이름이고 model destination name은 현재 implementation의 parameter 구조다. 둘이 같을 수도 있지만 fused projection, quant packing, tensor parallelism, expert mapping 때문에 다를 수 있다.

Q, K, V가 각각 `q_proj.weight`, `k_proj.weight`, `v_proj.weight`로 저장되고 destination은 `qkv_proj.weight` 하나라고 하자. name mapper는 세 source를 한 destination 이름으로 바꾸는 것에 그치지 않는다. 각 source가 destination output axis의 어느 slice에 들어갈지 shard ID를 함께 전달한다.

gate와 up projection도 fused `gate_up_proj`의 두 slices를 채울 수 있다. source order가 뒤바뀌면 shape는 같고 load는 성공하지만 activation 의미가 바뀐다. coverage는 destination tensor 전체가 채워졌다는 사실뿐 아니라 slice label이 맞다는 사실을 포함한다.

### 49.5.2 TP slicing

column-parallel weight는 output axis N을 TP ranks가 나누고 row-parallel은 input axis K를 나눌 수 있다. source full tensor에서 `start=tp_rank×shard_size`로 narrow하거나 storage format에 맞춘 custom loader를 쓴다.

logical shape `[K,N]=[4096,11008]`, TP=4 column split이면 ideal N shard는 2752다. kernel packing alignment 때문에 destination이 padded 2816일 수 있다. source slice 2752 columns를 destination original region에 copy하고 padded region을 zero/format-specific value로 채워야 한다. simple `param.copy_(loaded_weight)`는 맞지 않는다.

row split에서 quant group size 128과 local K boundary가 맞아야 scale/ZP slicing이 바르다. qweight packed dimension은 int32 words라 logical K offset에 packed factor를 적용한다. 46장의 Marlin mapping과 loader source가 만나는 지점이다.

### 49.5.3 EP filtering과 mapping

global expert IDs 0–63 중 rank가 `[16..23]`을 소유한다면 checkpoint `experts.16` source가 destination local expert 0에 들어갈 수 있다. source name parser, local expert set, destination index transformation이 하나의 rule이어야 한다.

replicated/shared expert가 있으면 simple contiguous range가 아닐 수 있다. EPLB placement가 expert mapping을 바꿀 수 있고 checkpoint는 canonical global order를 유지한다. early IO filter가 mapping보다 먼저 wrong set을 고르면 필요한 tensor를 읽지 않는다.

expert weight가 multiple shards에 split돼 있을 수도 있다. name 하나가 whole tensor라고 가정하지 않고 loader parameter metadata가 shard axis와 offset을 정한다. file shard와 tensor-parallel shard는 다른 개념이다. file shard는 serialization file 분할이고 TP shard는 tensor logical axis 분할이다.

### 49.5.4 quantized packed mapping

quant checkpoint는 qweight, qzeros, scales, g_idx 같은 여러 source tensors가 destination method의 parameters를 구성한다. process-after-loading이 repack하면 original storage와 converted storage가 일시적으로 겹칠 수 있다.

source header shape는 packed representation이다. `qweight` U32 shape `[K/8,N]`을 logical `[K,N]`과 직접 비교하면 shape mismatch로 오진한다. scalar type bits와 packed dimension을 사용해 logical shape를 복원한 뒤 destination Marlin tile shape를 계산한다.

conversion이 끝난 뒤 source parameter를 replace/delete하고 destination object를 등록하는 순서를 본다. model.named_parameters coverage가 converted names를 기준으로 하는지 source names를 기준으로 하는지도 확인한다. strict check가 quant model에서 어려운 이유다.

### 49.5.5 dtype와 device materialization

Safetensors header dtype은 stored bytes의 해석이다. model config dtype override는 destination compute/storage dtype을 정할 수 있다. source FP32 norm weight를 BF16 parameter로 cast할 수도 있고 quant qweight U8은 그대로 packed storage에 둘 수 있다.

dtype conversion을 모든 tensors에 일괄 적용하면 quant code와 scale metadata를 망친다. parameter loader가 `is_quantized`/custom weight loader를 통해 source dtype을 보존하거나 format-specific conversion을 해야 한다. floating parameter와 integer packed parameter를 구분한다.

device placement도 parameter별이다. device map이 CPU, GPU, disk offload를 나누고 TP rank destination GPU가 다르다. source slice를 CPU에서 만들고 GPU에 copy할지, file slice에서 device로 직접 materialize할지 path가 다르다. destination allocation 전에 meta parameter가 있을 수 있다.

copy 뒤 source handle을 닫아도 destination이 독립 storage를 소유하는지 확인한다. destination이 file-backed CPU view를 그대로 parameter로 채택했다면 mapping lifetime이 model lifetime까지 필요할 수 있다. caller가 언제 close하는지는 storage ownership contract에 달렸다.

## 49.6 coverage validation은 네 집합과 mapping edge로 한다

### 49.6.1 네 집합

첫 집합은 checkpoint file headers에 실제 존재하는 source names다. 둘째는 index가 선언한 source names다. 셋째는 mapping이 소비한 source names다. 넷째는 model에서 채워야 할 destination parameter와 slice다.

index names와 header names의 차이는 packaging 문제다. header에는 있는데 index에 없으면 unindexed extra이고, index에는 있는데 owning shard에 없으면 corrupt/mixed snapshot이다. mapping이 소비하지 않은 header source는 unexpected다. destination coverage가 부족하면 missing이다.

QKV처럼 three-to-one mapping이 있으므로 집합 cardinality만 비교하지 않는다. edge를 `(source_name, destination_name, destination_slice, transform)`으로 기록한다. destination slice interval의 union이 expected tensor를 덮고 overlap이 없는지 본다.

### 49.6.2 missing

required destination parameter가 전혀 쓰이지 않거나 일부 slice가 비면 missing이다. tied parameter는 canonical weight를 load한 뒤 tie operation으로 storage가 연결될 수 있으므로 tie 후 coverage를 판단할 수 있다. optional head나 model variant의 intentionally missing key는 explicit allow rule이 필요하다.

missing parameter를 random initialize하고 warning만 내는 library behavior가 있을 수 있다. inference server에서는 silent random weight가 위험하다. strict mode와 quant exception policy, loading report를 확인한다. missing이 허용된 경우도 reason과 exact pattern을 남긴다.

### 49.6.3 unexpected

checkpoint source가 mapping에 소비되지 않으면 unexpected다. optimizer/training-only state, rotary cached buffer, extra classification head처럼 inference model이 의도적으로 무시할 수 있다. broad substring filter는 새 required weight를 잘못 버릴 수 있으므로 exact pattern과 model class 조건을 둔다.

wrong revision은 많은 unexpected와 missing이 쌍으로 나타날 수 있다. rename 하나라면 mapping update 후보지만 architecture가 바뀌었으면 config/weight revision mismatch를 먼저 해결한다. key rename으로 shape를 억지로 맞추지 않는다.

### 49.6.4 duplicate와 overlap

같은 source name이 consolidated와 shard에 두 번 나오면 source duplicate다. 같은 source가 mapping loop에서 두 번 소비되면 consume duplicate다. 서로 다른 source가 destination 같은 slice를 덮으면 destination overlap이다.

Python dict `update`는 duplicate name을 last-wins로 숨긴다. shard마다 state dict를 merge하기 전에 `(name,file)` origin table을 만들고 duplicate면 fail한다. tied alias처럼 의도적 multiple names→same storage는 destination slice overwrite와 다르다. tie operation으로 표현한다.

QKV destination slice가 `[0,q_end)`, `[q_end,k_end)`, `[k_end,v_end)`로 정확히 인접하는지 본다. off-by-one gap 또는 overlap이 있으면 전체 destination shape는 filled처럼 보여도 일부 row가 잘못된다.

### 49.6.5 shape와 dtype

shape 비교에는 네 단계가 있다. header stored shape, logical format shape, TP/EP local slice shape, destination physical/padded shape다. 어느 단계에서 처음 달라지는지 기록한다. quant packed tensor는 stored와 logical shape가 의도적으로 다르다.

dtype도 header source dtype, conversion intermediate dtype, destination storage dtype, compute dtype을 분리한다. qweight U8을 BF16 destination으로 cast하는 것은 오류일 수 있고 FP32 dense weight를 BF16 parameter로 cast하는 것은 option에 따른 정상일 수 있다.

shape mismatch error에 source file, key, stored shape/dtype, transform, rank, destination slice와 expected physical shape를 포함하면 조사 시간이 크게 줄어든다. “size mismatch” 한 줄로 끝내지 않는다.

## 49.7 option 하나가 어느 상태를 바꾸는가

### 49.7.1 `use_safetensors`

이 option은 format preference 또는 requirement를 바꾼다. true면 safetensors file/index가 없을 때 error하고 unsafe pickle-like `.bin` fallback을 막을 수 있다. false면 `.bin` path를 고를 수 있다. `None` default는 framework search order를 따른다.

효과를 검증하려면 최종 selected archive/index filenames와 loader iterator type을 본다. directory에 safetensors와 bin이 둘 다 있을 때 option이 file set을 바꾼다. 이미 safetensors만 있다면 true/default의 downstream path가 같을 수 있다.

### 49.7.2 eager/disable mmap

whole-file read는 mapping/page fault behavior를 피하고 network/FUSE path에서 sequential read를 만들 수 있다. 그 대가로 shard bytes와 decoded tensor storage가 겹치는 host peak 후보가 있다. shard size와 in-flight count가 핵심 조절 변수다.

file 32개×1 GiB와 file 2개×16 GiB는 total size가 같아도 eager peak가 다르다. sequential one-shard path라면 largest shard가 peak를 지배한다. multithread four-worker path에서는 largest concurrent shards 합이 중요하다. sharding policy가 loading memory behavior를 만든다.

### 49.7.3 prefetch

prefetch는 tensor materialization 전에 file blocks를 읽어 page cache를 채운다. later safe-open access의 fault latency를 줄일 수 있다. total bytes가 available RAM과 경쟁하고 다른 process cache를 evict할 수 있다.

effective `should_prefetch`는 explicit option, filesystem type, checkpoint total size와 RAM threshold로 결정된다. option default가 항상 off 또는 on이라고 문서화하지 않는다. startup log의 FS/size/RAM과 code predicate를 연결한다.

### 49.7.4 worker와 buffer depth

workers는 동시 read/decode 수를 바꾸고 buffer depth는 ready result가 소비자를 앞설 수 있는 정도를 바꾼다. 둘은 같은 옵션이 아니다. workers=8이어도 queue depth=1이면 완료 shard가 무한히 쌓이지 않는다. unbounded futures면 workers보다 많은 result가 대기할 수 있다.

IO throughput 이득은 storage queue depth와 network latency, consumer GPU copy 속도에 달렸다. peak는 in-flight shard bytes, decoded state, current destination conversion의 합이다. failure 시 pending work cancellation도 효과의 일부다.

### 49.7.5 dtype/device/meta

meta loading은 header dtype/shape placeholder만 만들어 model 구조와 placement를 정한다. 실제 weight byte IO를 없애는 최종 mode가 아니라 materialization을 늦춘다. device map이 destination별 필요한 tensor를 slice/load한다.

dtype override는 floating source를 cast하여 destination bytes를 줄일 수 있지만 cast temporary와 accuracy가 있다. packed quant code에는 적용하지 않는다. disk offload는 GPU peak를 줄이는 대신 file layout/IO와 execution-time transfer 계약을 추가한다.

### 49.7.6 GGUF mmap/mlock/direct IO/offload

mmap on은 file-backed CPU view path를 가능하게 한다. mlock은 pages를 resident하게 유지하려 해 RAM pressure를 높일 수 있다. direct IO는 page cache를 우회하고 alignment-expanded reads와 staging buffers를 사용한다. GPU layer offload는 selected tensors를 device backend buffer로 copy한다.

옵션별 효과는 tensor placement plan과 file ownership을 붙여 설명한다. mmap을 켜도 GPU-offloaded weight는 device copy가 필요할 수 있고 CPU-resident quant tensors는 mapping을 직접 사용할 수 있다. 모든 tensors가 같은 path를 타지 않는다.

## 49.8 세 incident를 첫 divergence로 푼다

### 49.8.1 partial download

여덟 shard 중 세 번째가 없다는 첫 사건을 보자. 눈으로 본 filename 대신 resolved snapshot directory와 index content를 읽는다. `weight_map.values()` unique set과 actual files exact set을 비교한다. index가 `model-00003-of-00009`를 가리키는데 directory에는 `...of-00008`이 있다면 서로 다른 packaging revision이다.

file이 정말 없으면 loader가 header iteration 전에 fail해야 한다. 여러 worker가 이미 다른 shards를 device에 copy하기 전에 index fail-fast check를 둔다. download layer는 same resolved commit에서 missing blob을 다시 가져온다. directory의 비슷한 filename을 rename해 채우지 않는다.

file은 있지만 size가 remote manifest보다 작으면 truncated download다. safetensors 첫 8 byte N과 header는 정상일 수 있고 data end에서 실패할 수 있다. 각 tensor END의 maximum과 data buffer actual length를 비교하면 full tensor iteration 전에 잡을 수 있다.

### 49.8.2 corrupt header/offset

N이 absurd하게 크면 parser의 header size bound와 file size check에서 fail한다. JSON dtype/shape가 malformed이면 tensor를 만들기 전에 fail한다. header는 정상인데 one tensor offsets가 overlap/out-of-bounds면 해당 name과 range를 보고한다.

content hash가 있다면 parse 전에 corruption을 확정할 수 있다. hash가 없으면 structural validation은 bit flip이 valid float data 안에서 난 경우를 잡지 못한다. repository LFS/blob integrity와 download cache verification이 필요하다. format이 안전한 parsing을 제공한다는 것과 cryptographic model authenticity를 보장한다는 것을 구분한다.

GGUF corruption은 magic/version, metadata type, counts, alignment, tensor offset expected sequence에서 잡는다. quant block data bit flip은 structure가 여전히 valid할 수 있다. split별 hash manifest가 없다면 output validation이 마지막 방어가 된다.

### 49.8.3 mixed revision

config는 revision A, index는 B, shard3은 C라고 하자. filename과 tensor keys가 같으면 file existence와 header structure가 모두 통과할 수 있다. shape가 달라지면 destination load에서 드러나지만 shape까지 같고 values만 다른 model checkpoint이면 silent wrong output이 된다.

resolved commit manifest가 첫 방어다. local modifications를 허용하면 각 file content hash를 저장한다. startup report가 model ID/revision만 말하지 않고 effective snapshot commit과 optional override files를 기록해야 한다.

index hash와 shards hash를 하나의 atomic download completion marker에 묶는다. incomplete directory를 final cache path로 노출하지 않고 temporary snapshot에서 모든 required files를 검증한 뒤 publish한다. 여러 process가 같은 cache를 읽을 때 partial state를 보지 않게 한다.

### 49.8.4 wrong tensor mapping

file set과 header가 맞는데 `down_proj` shape mismatch가 난 둘째 사건을 따라가자. stored shape, quant logical shape, TP local slice, destination physical shape를 순서대로 적는다. stored `[K/8,N]` qweight를 dense `[K,N]`과 비교했다면 format interpretation 단계가 틀렸다.

TP rank/size가 checkpoint export 때와 다르더라도 full HF checkpoint는 loader가 새로 slice할 수 있다. 이미 TP-sharded checkpoint라면 rank file과 parallel config가 맞아야 한다. file shard 번호를 TP rank로 가정하지 않는다.

name mapping이 architecture rename을 놓쳤으면 missing/unexpected가 쌍으로 나온다. shape가 우연히 같은 다른 destination으로 mapping되면 더 위험하다. mapping table에는 expected semantic shard label과 destination interval을 둔다.

### 49.8.5 duplicate last-wins

세 번째 사건의 silent wrong output에서 source origin table을 본다. 같은 key가 consolidated와 shard에 존재하고 merged dict update가 뒤 값을 선택했을 수 있다. load order를 바꾸면 output hash가 바뀌는 것이 강한 신호다.

각 yield에 origin file과 header offset을 붙여 duplicate source name에서 즉시 fail한다. destination copy에도 source origin과 slice interval을 기록한다. duplicate가 tied alias인지 판단하려면 model tie rule을 보고, file 중복을 alias라고 합리화하지 않는다.

### 49.8.6 bounded failure cleanup

worker 하나가 corrupt shard에서 실패하면 main iterator가 exception을 받고 pending reads를 취소한다. open handles, bytes dictionaries, pinned buffers, partially built model parameters를 해제한다. GPU destination 일부가 채워진 model object를 serving registry에 publish하지 않는다.

재시도는 새 model instance 또는 명확히 reset된 destination에서 시작한다. workspace/parameters 일부만 덮어쓰면 이전 attempt byte가 missing slice에 남아 validation을 통과할 수 있다. load completion은 coverage proof 뒤 atomic하게 표시한다.

## 49.9 LOAD-49: 파일은 맞았지만 rank 1 parameter는 절반만 새 세대였다

### 49.9.1 작은 Safetensors tensor를 absolute byte까지 내린다

fixture weight는 `model.layers.0.mlp.down_proj.weight`, shape `[N=8,K=4]`, FP16이다. elements는 32개, payload는 64 bytes다. file 첫 8 bytes가 little-endian header length `H=248`을 담고 JSON header가 뒤따른다고 하자. payload base는 `8+248=256`이다.

header의 `data_offsets=[64,128]`은 payload base 상대 범위라고 하자. tensor absolute file range는 `[320,384)`다. element `(n=5,k=2)`의 row-major element index는 `5×4+2=22`, tensor-relative byte offset 44, absolute byte offset 364다. FP16 두 bytes `[364,366)`를 읽는다.

range reader가 `data_offsets[0]=64`를 file absolute로 해석하면 JSON header 안을 tensor payload로 읽는다. 길이는 여전히 64 bytes라 shape/byte-count 검사가 통과할 수 있다. random header bytes를 FP16로 해석해 NaN 또는 plausible values가 섞인다. offset base identity를 metadata와 함께 검증한다.

little-endian header length도 명시한다. bytes를 host-native integer로 무조건 cast하지 않고 format contract에 따라 decode한다. huge H, file bounds 초과, JSON parse failure를 payload mmap 전에 막는다. tensor dtype payload의 byte order와 quant block field conventions도 format별 source로 확인한다.

### 49.9.2 index→shard→slice→destination을 한 행으로 잇는다

model index는 tensor name을 `model-00002-of-00004.safetensors`에 매핑한다. revision commit R17에서 index와 shard filename, expected size/hash를 고정한다. filename이 존재한다는 사실만으로 R17 content라고 판단하지 않는다. local cache의 blob identity와 revision snapshot symlink를 연결한다.

TP=2가 N축 row-wise shard를 요구하면 rank 0 destination은 source rows `[0,4)`, rank 1은 `[4,8)`이다. rank 1 slice tensor-relative byte range는 rows 네 개×K4×2=32 bytes이므로 `[32,64)`, file absolute `[352,384)`다. sample `(5,2)`는 rank1 local row 1, local element index 6, destination byte offset 12다.

loader가 full tensor를 materialize한 뒤 slice할 수도 있고 safetensors slice/range로 필요한 rows만 읽을 수도 있다. 두 paths는 peak memory와 I/O가 다르지만 destination values는 같아야 한다. source slice semantics가 dimension/stride를 보존하는지 확인한다. packed quant tensor는 logical N slice가 raw byte contiguous가 아닐 수 있다.

destination parameter identity는 `(model_load_tx=49, rank=1, parameter_name, generation=17)`로 둔다. CPU slice가 만들어졌다는 사실과 GPU parameter storage에 copy 완료된 사실, model registry에 generation 17이 publish된 사실을 나눈다. load iterator yield를 commit으로 읽지 않는다.

partial commit 사건은 같은 fixture의 수치 timeline으로 재구성한다.

old model generation 16이 serving 중이다. loader는 generation 17을 staging model에 적재한다. t0에 index R17을 읽고 t1에 shard 1을 검증한다. t2에 rank0 down_proj rows 0–3 copy가 끝난다. t3에 rank1이 shard 2 range `[352,384)`를 읽지만 remote object가 16 bytes 뒤 끊긴다.

버그 난 coordinator는 parameter별 `loaded_names`에 down_proj를 rank0 성공 시 추가하고 registry pointer 일부를 새 storage로 바꿨다. rank1 실패 뒤 cleanup는 rank1 staging만 버렸다. 다음 request는 rank0 generation17 rows와 rank1 generation16 rows를 collective GEMM에서 함께 사용했다. model name/revision label은 R17 하나로 보였다.

correct transaction은 rank별/local parameter copy를 staging에 유지하고 모든 required coverage, shape/dtype, device copy completion, distributed acknowledgment가 닫힌 뒤 model generation을 atomic publish한다. 한 rank 실패면 generation17 전체를 abort하고 generation16 registry를 유지한다. 이미 enqueue된 device copies는 completion 뒤 staging storage를 회수한다.

loaded-name tracking은 coverage evidence이지 publish authority가 아니다. name이 set에 있다는 사실은 어느 rank slice가 어느 generation destination에 commit됐는지 말하지 않는다. `(source tensor, source slice, destination rank/parameter, copy completion)` edges를 센다.

GGUF에서는 tensor directory와 quant block을 같이 검산한다.

GGUF fixture tensor는 logical `[N=8,K=32]`, quant block이 32 values마다 scale FP16 2 bytes와 quants 16 bytes를 가진다고 가정한다. row 하나가 block 하나라 physical bytes는 18, tensor 전체 144 bytes다. 이 수치는 특정 실제 type의 보편 공식이 아니라 교육용 contract이며 실제 GGUF type trait를 pinned source에서 가져와야 한다.

directory가 tensor offset 1024를 주고 alignment 32라면 payload start가 실제 file rules와 맞는지 확인한다. logical `(n=5,k=7)`은 block row 5, block base `1024+5×18=1114`다. scale은 `[1114,1116)`, 4-bit pair byte는 `1116+floor(7/2)=1119`, nibble 선택은 quant type convention에 따른다.

TP N-slice rank1 rows 4–7은 logical values 128개지만 raw range는 four complete quant blocks 72 bytes `[1096,1168)`다. block-aligned 축이면 direct range가 가능하다. K축을 16/16으로 나누면 block을 둘이 공유하므로 raw byte 반쪽 slice가 곧 독립 quant tensor가 아니다. dequant/special shard contract가 필요하다.

GGUF metadata의 architecture, tensor type, dimensions, alignment, split-file fields가 filename보다 중요하다. 같은 basename과 revision label이어도 directory offsets나 quant type가 다르면 consumer trait가 달라진다. llama.cpp loader가 type traits와 tensor directory를 어떻게 소비하는지 함께 pin한다.

Transformers·vLLM·llama.cpp source는 commit까지 이어서 걷는다.

Transformers에서는 revision/commit resolution, index weight map, shard fetch/cache, safetensors open/slice, parameter loading report를 잇는다. missing/unexpected/mismatched keys report가 언제 만들어지고 model object가 caller에게 반환되는지 본다. low-memory/meta-device path는 destination materialization 경계가 다를 수 있다.

vLLM에서는 load format selection, weight iterator, model-specific weight loader, TP/EP slice consumer, loaded-name validation을 잇는다. iterator가 yield한 full tensor를 destination parameter loader가 어떤 axis와 packed metadata로 자르는지 sample `(5,2)`로 확인한다. quantized parameter는 일반 dense slice 규칙을 그대로 적용하지 않는다.

llama.cpp에서는 GGUF header/KV/tensor directory parse, mmap/direct read selection, type trait, backend tensor allocation/copy, device offload commit를 잇는다. mmap pointer lifetime과 model context lifetime을 연결한다. split GGUF라면 모든 files의 metadata consistency와 tensor ownership을 검증한다.

세 스택을 같은 state-dict API로 설명하지 않는다. 공통 질문은 identity resolution, directory/index ownership, byte range validation, logical-to-physical slice, destination allocation, coverage, publish/rollback이다. 구체 자료구조와 lazy/eager 전략은 구현별로 보존한다.

마지막으로 반증·rollback·terminal을 닫는다.

첫 fixture는 Safetensors header relative/absolute offset을 일부러 바꾼다. expected sample `(5,2)` sentinel이 file offset 364에서만 읽히는지 본다. byte length가 맞아도 header 영역을 읽은 path는 content hash/basis에서 실패해야 한다.

둘째는 range truncation이다. rank1 `[352,384)` 요청에 16 bytes만 반환하고 short read를 성공으로 취급하지 않는지 본다. destination staging의 unwritten half에는 poison을 두고 publish가 금지되는지 확인한다. retry가 다른 revision blob과 섞이지 않게 object identity를 고정한다.

셋째는 mixed revision이다. R17 index가 shard filename을 가리키지만 local cache에 R16 blob만 있을 때 size가 우연히 같아도 hash/commit identity로 거부한다. index와 모든 shards가 같은 snapshot graph에 속해야 한다.

넷째는 distributed partial failure다. rank0 copy success 뒤 rank1을 실패시키고 active registry가 generation16을 유지하는지 본다. rank0 staging17은 request에 보이지 않아야 하며 device work completion 뒤 회수된다. 모든 ranks success 때만 generation17을 atomic publish한다.

다섯째는 GGUF endian/quant trait다. known block scale와 nibbles를 넣고 logical row basis를 dequant한다. wrong byte order, nibble order, block size가 각각 distinct signature로 실패해야 한다. file size/offset bounds만으로 quant meaning을 검증했다고 하지 않는다.

rollback는 active generation16을 유지하고 failed generation17 staging을 격리한다. corrupt/mixed cache blobs는 forensic hash와 failing range를 기록한 뒤 namespace에서 제외한다. current pinned revision에서 shards를 다시 받아 full manifest 검증 뒤 load transaction을 재시도한다.

90분 soak는 safetensors eager/lazy/range, GGUF mmap/direct, TP1/2, cache cold/warm, short read/retry를 섞는다. tensor coverage, rank generation equality, poison unwritten, hash mismatch, mmap lifetime 오류가 0이어야 한다. peak RSS/VRAM과 load time은 correctness 통과 뒤 비교한다.

terminal 문장은 구체적이다. “R17 shard2의 rank1 range `[352,384)`가 16 bytes short였는데 rank0 parameter publish가 먼저 일어나 generation16/17이 섞였다.” fix는 manifest-bound range validation, staging-only load, all-rank coverage barrier, atomic model publish로 증명한다.

## 49.10 실습: tensor 하나의 load transaction을 왕복한다

먼저 tutorial 원장과 reference 원장을 분리한다.

tutorial 원장은 독자가 실행할 순서를 담는다. revision pin, manifest/index resolve, header parse, range validation, logical slice, destination copy, coverage barrier, atomic publish, rollback 순서다. 각 단계에는 fixture 숫자와 expected result를 둔다.

reference 원장은 format/library 사실을 담는다. Safetensors header length encoding과 relative data offsets, GGUF version/metadata/directory/alignment/type traits, Transformers revision resolution, vLLM iterator/parameter consumer, llama.cpp mmap/offload lifecycle의 고정 source다. tutorial의 교육용 GGUF block 공식을 실제 type trait라고 섞지 않는다.

두 원장은 evidence ID로 연결한다. tutorial에서 `payload_base=8+H`를 계산할 때 Safetensors format parser source를 참조한다. GGUF block 18-byte 예제는 “fixture contract”로 표시하고 실제 Q4 계열 type을 조사할 때 pinned trait 값으로 교체한다. 독자는 계산 절차와 현재 구현 사실을 구분할 수 있다.

1단계는 revision과 file graph를 봉인하는 일이다.

load transaction L49는 model repo commit R17을 입력으로 받는다. index JSON bytes/hash, shard filenames, 각 blob identity/size/hash, config/tokenizer 관련 identity를 manifest에 둔다. `main`이나 filename만 기록하지 않는다. local cache path는 content identity가 아니라 retrieval location이다.

index weight map에서 down_proj가 shard2를 소유한다는 edge를 만든다. 같은 tensor가 두 shards에 나타나거나 index가 가리키지 않는 consolidated file이 함께 발견되면 policy에 따라 fail/ignore하되 선택을 기록한다. directory glob order에 last-wins를 맡기지 않는다.

remote range read를 쓴다면 ETag/object version과 expected blob hash를 transaction에 묶는다. retry가 다른 revision object로 넘어가지 않게 한다. `206 Partial Content` status만으로 requested byte count와 content-range identity를 대신하지 않는다.

2단계에서는 byte range를 bounds와 content로 검증한다.

file size를 F라고 할 때 header length H는 `8+H≤F`를 만족해야 한다. 각 tensor relative range `[a,b)`는 `0≤a≤b≤F-(8+H)`이고 absolute `[8+H+a,8+H+b)`다. expected bytes는 dtype element size×shape product 또는 quant type physical size와 일치해야 한다.

fixture tensor `[8,4]` FP16의 expected bytes는 64다. `[64,128)` 길이도 64이고 absolute `[320,384)`도 file bounds 안이다. range reader는 정확히 64 bytes를 반환해야 한다. short 16/32/63 bytes를 EOF success로 취급하지 않는다.

content probe는 row별 sentinel을 쓴다. row n의 first value를 `100+n`으로 만들면 rank1 range rows4–7의 first values가 104,105,106,107이어야 한다. bytes가 길이만 맞고 offset base가 틀리면 signature가 실패한다. cryptographic blob hash와 coordinate probe는 서로 다른 오류를 잡는다.

3단계에서는 logical slice와 packed slice를 구분한다.

dense FP16 N-axis slice는 contiguous rows라 direct range가 가능하다. rank1 `[4:8,:]`은 tensor-relative `[32,64)`. K-axis slice `[ :,2:4]`는 각 row의 일부라 하나의 contiguous range가 아니다. stride-aware slices 또는 full read가 필요하다.

quantized tensor에서는 logical axis와 block packing이 더 갈라진다. block boundary에 맞는 N rows는 contiguous할 수 있지만 K shard가 quant block을 자르면 scale/header를 공유한다. parameter loader가 packed shard를 이해하는지, conversion 후 slice하는지, format-specific sharder를 쓰는지 source로 확인한다.

TP destination shape도 검산한다. global `[8,4]`, N-shard TP2면 local `[4,4]`다. parameter storage가 transposed `[K,Nlocal]` 또는 packed form을 기대하면 destination loader가 변환을 소유한다. source slice shape만 맞다고 copy contract가 맞는 것은 아니다.

4단계에서는 staging copy와 coverage barrier를 닫는다.

rank마다 staging model generation17을 만든다. parameter copy record는 source blob/range, logical slice, destination name/range, dtype/format, copy completion을 가진다. GPU async copy를 enqueue한 host return가 completion은 아니다. publish barrier는 destination readiness를 확인한다.

coverage set은 expected destination slices와 loaded edges를 비교한다. rank0 rows0–3, rank1 rows4–7이 union global rows0–7을 정확히 한 번 덮어야 한다. 합계 elements 32만 맞으면 overlap/missing이 상쇄될 수 있으므로 interval identity를 본다.

tied/shared parameters는 deliberate alias edge로 표현한다. duplicate load와 shared storage를 구분한다. quant auxiliary tensors scales/zeros도 main qweight와 같은 parameter family coverage에 포함한다. qweight만 loaded-name set에 있으면 complete라고 하지 않는다.

all-rank barrier는 단순 process 도착이 아니라 manifest identity, expected/loaded coverage, copy completion, no validation errors를 모은다. 한 rank가 R16 manifest로 도착하면 count는 맞아도 publish를 거부한다. coordinator decision generation을 남긴다.

5단계에서는 atomic publish와 rollback를 관측한다.

publish 전 active model pointer는 generation16이다. 모든 ranks ready면 coordinator가 generation17 commit을 결정하고 request admission registry를 새 replica generation으로 전환한다. in-flight generation16 requests는 old model을 drain하거나 deployment contract에 따라 새 admission만 17로 보낸다.

rank별 registry를 독립 갱신하면 mixed-generation window가 생긴다. distributed serving에서는 router가 replica ready를 all-rank commit 뒤에만 노출해야 한다. rank0 publish success 로그 하나로 model ready metric을 올리지 않는다.

failure rollback는 transaction L49를 abort하고 staging allocations/handles/range buffers를 owner별로 정리한다. mmap slice view가 남았으면 file handle/mapping을 먼저 닫지 않는다. async device copy가 in-flight면 completion 뒤 storage를 회수한다. active generation16은 변경하지 않는다.

GGUF 실습은 directory owner에서 시작한다.

GGUF는 external index JSON 대신 file header/metadata/tensor directory가 tensor owner를 알려 준다. split files이면 split metadata와 filenames/indices가 하나의 set인지 검증한다. 모든 parts를 모으기 전에 tensor allocation를 publish하지 않는다.

directory entry는 name, dimensions, type, offset을 제공하고 data section alignment/base와 합쳐 absolute range를 만든다. offset 의미를 source parser와 spec에서 확인한다. tensor entries가 overlap하거나 bounds를 넘으면 mmap pointer를 backend에 넘기기 전에 fail한다.

quant physical size는 type trait의 block size와 type size로 계산한다. logical element count가 block size로 나누어지지 않을 때 padding/row rules를 따른다. 교육용 32→18 fixture의 결과를 실제 GGUF type 전반에 적용하지 않는다.

llama.cpp backend offload는 mapped file bytes가 CPU tensor storage로 남는지 GPU buffer로 copy되는지에 따라 mmap lifetime가 달라진다. model context가 mappings와 backend buffers를 소유하고 failure cleanup가 partial tensors를 해제하는 순서를 본다.

source walk의 종료 조건도 명시한다.

Transformers source walk는 revision resolution과 shard selection에서 parameter materialization/adjusted loading report, model return까지 이어져야 한다. report가 warning만 내는 mismatch와 fatal mismatch policy를 구분한다. `ignore_mismatched_sizes` 같은 option이 destination initialization에 어떤 상태를 남기는지 확인한다.

vLLM source walk는 iterator yield에서 끝나지 않는다. model-specific `load_weights`/parameter loader가 name conversion, TP/EP slicing, packed quant mapping, loaded-name validation을 수행하는 소비 지점까지 간다. final model runner가 load 성공 뒤 publish되는 boundary를 찾는다.

llama.cpp source walk는 GGUF parser에서 tensor allocation, mmap/read, backend upload, model/context successful return와 failure destructor까지 잇는다. split validation와 type trait consumer를 함께 pin한다. CLI filename parsing만으로 format 설명을 끝내지 않는다.

종료 표에는 identity, absolute range, logical coordinate, destination coordinate, completion, coverage, publish generation의 일곱 열이 모두 채워져야 한다. 빈 열은 추정으로 표시하고 load 완료를 승인하지 않는다.

마지막으로 incident 회귀 matrix와 운영 terminal을 닫는다.

matrix 축은 format safetensors/GGUF, path mmap/eager/range/direct, cache cold/warm, TP1/2, full/short/mixed revision, quant/dense다. 모든 조합을 무작정 곱하지 않고 각 state transition가 달라지는 대표 fixture를 선택한다.

short-read fixture는 poison unwritten 영역과 exact length check를 요구한다. mixed-revision fixture는 same filename/size지만 다른 hash를 쓴다. endian fixture는 header length와 known FP16/quant metadata field를 검산한다. offset fixture는 relative/absolute confusion을 유도한다.

partial-rank fixture는 rank0 copy 완료와 rank1 실패를 만든다. model-ready metric, router registry, active generation가 모두 16에 남아야 한다. staging17은 completion-safe cleanup 뒤 0이 된다. retry L50은 새 transaction generation을 사용한다.

정상 fixture에서 rank0/1 slices union은 full tensor reference와 같다. sample `(5,2)`는 file364→rank1 local12→destination value sentinel을 유지한다. GGUF basis도 directory offset→block scale/nibble→logical value→destination을 왕복한다.

운영 telemetry는 load transaction, pinned revision, manifest hash, shard/range progress, expected/loaded coverage, rank readiness, publish generation, rollback reason을 담는다. tensor values나 model secrets를 로그에 남기지 않고 hashes와 coordinate probes를 사용한다.

terminal은 generation17 publish 뒤 synthetic inference/reference, all-rank generation equality, coverage exactness, stale staging 0을 확인한다. 90분 reload soak에서 cache cold/warm과 failure injection를 반복하고 mixed generation, short-read success, mmap UAF가 0이어야 한다.

incident 회고 첫 문장은 “shard 파일이 깨졌다”가 아니다. “R17 shard2 `[352,384)` exact range가 16 bytes short였는데 rank0 staging slice가 active registry에 조기 publish됐다.” source fix와 regression fixture가 이 두 결함을 각각 닫아야 한다.

range retry 실습은 세 응답을 구분한다. 첫 요청 `[352,384)`가 16 bytes만 주고 끊기면 loader는 received interval `[352,368)`을 transaction ledger에 임시 기록한다. 재시도가 `[368,384)`를 이어 받는 방식이라면 두 responses가 동일 object version R17-B2인지 확인한다. object가 바뀌면 조각을 합치지 않고 처음부터 다시 받는다.

서버가 range를 무시하고 full object를 `200`으로 반환할 수도 있다. client가 body 첫 32 bytes를 requested slice로 오인하지 않게 content-range/status contract를 확인한다. full body를 수용할 정책이면 absolute `[352,384)`를 다시 slice하고 object hash를 검증한다. status code만 보고 pointer arithmetic를 하지 않는다.

HTTP 압축이나 transport encoding이 있으면 range offsets가 어느 representation을 기준으로 하는지도 확인한다. model blob storage는 보통 exact bytes identity가 중요하므로 transparent transformation를 피하거나 decoded content identity를 명확히 한다. remote layer 사실을 file-format offset과 섞지 않는다.

local cache partial file도 owner state를 가진다. download-in-progress temp path와 verified blob path를 분리하고 hash/size 검증 뒤 atomic rename/publish한다. 다른 loader가 temp file을 valid shard로 열지 못하게 한다. crash recovery는 incomplete temp를 재개하거나 폐기하되 verified namespace에 올리지 않는다.

mmap 실습에서는 mapping base, file descriptor, tensor view, backend copy의 수명을 그린다. lazy tensor view가 absolute `[320,384)`를 가리키는 동안 mapping을 닫으면 다음 slice access가 실패한다. GPU copy enqueue 뒤 host mapping을 바로 닫아도 binding/driver가 source bytes를 더 필요로 하는지 copy completion contract를 확인한다.

Transformers lazy path가 tensor를 materialize하는 시점과 Python context manager가 file handle을 닫는 시점을 source에서 맞춘다. `safe_open` block 밖으로 lazy slice object를 넘길 수 있는지 API contract가 결정한다. eager tensor가 독립 storage를 가졌다면 mapping lifetime가 다르다. 이름이 tensor라고 lifetime를 같게 보지 않는다.

vLLM iterator가 shard handle 안에서 tensor를 yield하고 consumer가 즉시 parameter copy를 끝내는지, yield object를 queue/prefetch에 보존하는지 본다. threaded/buffered loading은 producer context가 닫히기 전에 consumers completion을 보장해야 한다. queue put 성공은 GPU copy 완료와 다르다.

TP dense slice를 address 표로 확장한다. source rows4–7의 각 row는 8 bytes다. rank1 local rows0–3의 destination offsets는 0,8,16,24다. source absolute offsets는 352,360,368,376이다. 네 row mapping을 set으로 비교하면 stride 8 대신 4를 쓴 overlap bug를 잡을 수 있다.

destination parameter가 transposed `[K,Nlocal]=[4,4]`를 기대하면 source `(n=5,k=2)`는 local `(k=2,nlocal=1)`, element index `2×4+1=9`, byte offset18이다. 앞의 row-major destination offset12와 다르다. loader의 destination layout contract를 확인하지 않고 memcpy하지 않는다.

column-parallel/row-parallel 이름도 축을 자동으로 증명하지 않는다. framework parameter loader가 weight transpose conventions와 fused QKV/gate-up partitions를 가질 수 있다. model-specific mapping 함수가 source tensor를 여러 destination slices로 나누거나 여러 source tensors를 하나로 합치는지 edge로 기록한다.

fused QKV example은 source Q/K/V 각각 `[8,4]`가 destination `[24,4]`로 concat된다고 하자. tensor coverage는 source names 세 개와 destination intervals `[0,8)`, `[8,16)`, `[16,24)`를 연결한다. loaded-name count 3만 맞아도 interval order가 Q,V,K로 잘못될 수 있다. coordinate sentinel로 순서를 검증한다.

quant packed TP slice는 physical blocks의 owner를 센다. K block 32를 TP2로 16씩 나누려면 두 ranks가 같은 block scale을 공유하거나 unpack/requantize가 필요할 수 있다. raw 18-byte block을 9 bytes씩 나누는 것은 nibble/scale contract를 보존하지 않는다. format-specific loader가 reject 또는 변환하도록 한다.

GGUF tensor dimensions order도 model logical convention과 parser representation을 대조한다. directory의 dimension array를 `[K,N]`으로 읽는지 `[N,K]`로 읽는지 type/loader contract를 본다. element count는 곱이라 같아 transpose 오류를 shape-product 검사로 잡지 못한다. basis row/column signature가 필요하다.

GGUF split set은 part count, part index, tensor count/ownership, metadata consistency를 검증한다. part 1은 R17이고 part 2가 R16인데 filenames만 맞으면 tensor directory union이 plausible할 수 있다. split-set UUID/hash 또는 pinned manifest identity로 혼합을 거부한다.

directory offsets는 data section base와 alignment padding을 포함해 absolute로 계산한다. tensor A 끝과 tensor B start 사이 hole은 alignment로 정상일 수 있다. overlap은 alias가 명시된 format contract가 없다면 오류다. hole bytes를 tensor payload로 세지 않는다.

endianness fixture는 header fields와 payload values를 분리한다. file magic/version/count/offset integers의 byte order, FP16 scale bits, quant code nibble order가 각각 있다. host architecture가 little-endian이라는 사실로 모든 subfield convention을 추정하지 않는다. pinned parser/type trait가 evidence다.

model parameter commit에는 initialized sentinel을 둔다. staging allocation 직후 poison하고 성공 copy가 expected destination interval 전체를 덮으면 poison이 사라져야 한다. padding/unowned intervals는 별도 expected sentinel로 남을 수 있다. publish 전에 poison scan 또는 coverage proof를 수행한다.

asynchronous GPU copy를 쓰면 copy event generation을 parameter edge에 붙인다. rank ready acknowledgment는 모든 parameter copy events complete 뒤에 보낸다. host enqueue가 끝난 뒤 즉시 ready를 보내면 registry publish와 device write가 겹친다. load correctness에도 stream lifetime 원칙이 적용된다.

all-rank coordinator는 two-phase 형태로 이해할 수 있다. prepare 단계에서 각 rank가 manifest/coverage/completion digest를 보낸다. commit 단계에서 동일 transaction generation에 모두 ready일 때 registry switch를 지시한다. 한 rank abort면 전 ranks가 staging generation을 discard한다.

commit message가 일부 ranks에만 도착한 crash도 고려한다. router가 replica를 외부 ready로 만들기 전에 ranks generation handshake를 확인한다. 불확실하면 replica 전체를 traffic에서 빼고 재시작/재로드한다. 서로 다른 model generations의 collectives를 실행하지 않는다.

rollback cleanup 순서는 iterator/tasks 취소, 신규 reads 중단, outstanding copies drain, destination staging free, mmap/handles close, temp cache cleanup다. source view가 남았는데 mapping을 먼저 닫거나 device copy 중 staging을 free하지 않는다. owner ledger로 역순을 결정한다.

retry는 L49 transaction ID를 재사용하지 않고 L50을 만든다. same pinned revision을 쓸 수는 있지만 staging allocations, range responses, copy events, rank acknowledgments는 새 generation이다. late L49 callback가 L50 readiness를 변경하지 못하게 한다.

Transformers tutorial 검증은 returned model parameter sample과 loading report를 함께 본다. warning를 허용하는 option이 있으면 unwritten/mismatched destination이 어떻게 initialized됐는지 설명한다. serving correctness 책에서는 편의상 ignore한 mismatch를 load success로 숨기지 않는다.

vLLM tutorial 검증은 model-specific loader가 반환/기록한 loaded parameter names와 expected model parameters를 비교하고, TP rank sample values를 reference full tensor slice와 대조한다. skipped expert/stacked parameter가 deliberate mapping인지 gap인지 mapping edges로 판정한다.

llama.cpp tutorial 검증은 GGUF metadata dump만 보지 않고 backend tensor sample과 offload placement를 확인한다. CPU mmap tensor와 GPU buffer가 어느 layers를 소유하는지, partial offload 실패가 model context publish를 막는지 본다. split parts handle lifetime도 context와 연결한다.

관측 비용은 load path에 맞춘다. 모든 tensor full hash가 비싸면 blob cryptographic hash와 selected coordinate probes, directory validation를 조합한다. 그러나 remote/cache 공급망이 신뢰할 수 없는 경계에서는 full content identity가 필요할 수 있다. 성능 때문에 revision identity를 filename으로 낮추지 않는다.

load progress metric은 bytes downloaded, bytes verified, tensors parsed, destination slices copied, ranks prepared, model generation committed를 분리한다. bytes 100%가 model ready는 아니다. quant conversion/repack과 GPU copy, distributed barrier가 뒤에 남을 수 있다.

failure labels는 `revision_mismatch`, `missing_shard`, `short_range`, `header_bounds`, `tensor_overlap`, `slice_mapping`, `copy_completion`, `rank_prepare`, `publish`처럼 first divergence를 드러낸다. `load_error` 하나로 재시도 policy를 고르지 않는다.

retryable transport short read와 non-retryable format bounds corruption을 구분한다. 동일 object에서 exact range를 다시 받을 수 있으면 retry하되 반복 hash mismatch는 cache/object를 격리한다. mapping/shape mismatch를 네트워크 retry로 숨기지 않는다.

canary reload는 active generation16 traffic를 유지하며 staging17을 만들고 shadow probe를 실행한다. all-rank values와 synthetic logits가 reference를 통과한 뒤 새 admission만 17로 전환한다. generation16 in-flight drain 뒤 old model/storage를 해제한다.

rollback 후에도 registry generation, rank generations, mmap count, open file handles, staging GPU bytes, temp downloads가 baseline으로 돌아왔는지 본다. file descriptor leak나 orphan staging는 다음 reload failure로 이어질 수 있다. correctness와 resource cleanup를 함께 닫는다.

최종 reference section에는 format spec/source commit, parser/loader source spans, model revision manifest, fixture calculations를 분리해 링크한다. tutorial prose는 독자가 따라갈 순서를 제공하고 reference는 현재 버전 사실을 검증한다. 버전이 바뀌면 reference를 갱신하되 load transaction 질문은 유지한다.

terminal 승인문은 sample을 포함한다. “R17 shard2 tensor `[320,384)`에서 TP rank1 `[352,384)`를 exact-read하고 `(5,2)` file364를 local destination layout의 올바른 offset에 copy했다. 모든 ranks generation17 coverage/completion barrier 뒤 registry를 atomic publish했고 L49 fault fixtures에서 generation16을 보존했다.” 이 문장을 source와 trace로 재현해야 한다.

이 원장이 완성되면 filename, mmap, state dict, GGUF 같은 단어가 load success의 대리 지표가 되지 않는다. identity에서 bytes, logical coordinates, destination, completion, distributed publish까지 이어지는 transaction만이 serving 가능한 model generation을 만든다.

실전 source drill은 tensor name 하나로 시작한다. index나 GGUF directory에서 owner file을 찾고 parser가 만드는 offset/shape/type object를 따라간다. 그 object가 iterator에 들어가고 model-specific name mapper를 지나 destination parameter loader에 도착하는 모든 함수의 입력·출력을 한 줄씩 적는다. repo 전체 파일 목록을 만드는 대신 이 한 tensor의 실제 consumer path를 닫는다.

Transformers에서는 `_from_pretrained` 계열의 고수준 호출만으로 끝내지 않는다. checkpoint file resolution 결과, shard iteration, safetensors handle/slice, meta parameter materialization, load-state report가 어느 branch에서 이어지는지 본다. device map/offload가 있으면 destination owner가 CPU/disk/GPU 가운데 어디인지 추가한다.

vLLM에서는 default loader와 dummy/tensorizer/other formats를 먼저 dispatch에서 구분한다. safetensors iterator strategy normal/eager/prefetch가 동일 parameter consumer로 수렴하는지 확인한다. prefetch가 있으면 bounded queue, exception propagation, handle lifetime, cancellation cleanup를 본다.

vLLM model-specific `weight_loader`는 stacked parameters와 packed quant parameters를 처리할 수 있다. source name 하나가 QKV destination의 offset으로 가거나 gate/up projection slice로 들어갈 수 있다. tensor name equality만으로 coverage를 계산하지 않고 mapping edge와 destination interval을 센다.

EP early skip은 누락과 다르다. rank가 소유하지 않는 expert tensor를 읽기 전에 skip하면 expected set에서 제외되는 predicate가 있어야 한다. global manifest coverage와 rank-local destination coverage를 둘 다 둔다. 모든 rank local sets의 union이 intended global experts를 덮는지 본다.

llama.cpp에서는 model architecture metadata가 tensor-name expectations를 만들고 GGUF directory entries를 destination tensors와 연결한다. missing required tensor, duplicate name, unexpected type/dimensions가 model context publish 전에 실패하는지 확인한다. optional tensors와 architecture variants는 source predicate로 구분한다.

GGUF mmap은 file-backed CPU access를 효율화하지만 GPU offload는 device buffer allocation와 transfer를 추가한다. `mmap=true`라는 option만으로 GPU memory peak나 copy completion을 설명하지 않는다. mapped bytes residency, backend buffer bytes, staging/copy overlap을 각각 측정한다.

direct read는 destination buffer alignment와 read granularity가 성능을 바꿀 수 있다. correctness 원장은 requested/received exact ranges와 destination offsets를 유지한다. optimization으로 ranges를 coalesce하면 각 tensor subrange가 merged response 안에서 올바른 offset을 갖는지 다시 검산한다.

coalesced example은 tensor A `[320,384)`, padding hole `[384,416)`, tensor B `[416,480)`를 한 번에 `[320,480)` 읽는 경우다. A local offset0, B local offset96이다. B를 A length64 직후 offset64로 두면 hole을 무시해 32 bytes 앞당겨진다. directory absolute offsets를 유지한다.

safetensors 여러 tensors도 range coalescing에서 header relative offsets를 먼저 absolute로 바꾼 뒤 정렬/병합한다. overlap detection를 병합 전에 수행한다. aliases가 허용되지 않는 format에서 동일 range를 두 names가 가리키면 duplicate ownership 오류로 처리한다.

loader concurrency는 memory peak와 failure blast radius를 바꾼다. threads 8개가 shards를 eager load하면 eight shard buffers와 destination copies가 겹칠 수 있다. bounded prefetch depth2면 peak를 줄이고 오류 이후 outstanding work도 제한한다. throughput만 보고 unbounded submit하지 않는다.

exception propagation는 producer thread 오류를 consumer가 최종 성공으로 덮지 않게 한다. 이미 yielded tensors가 staging에 copy됐어도 iterator terminal error가 있으면 transaction abort다. queue sentinel와 exception object를 구분하고 join 뒤 모든 workers가 닫혔는지 확인한다.

revision race도 재현한다. load 시작 때 `main`이 R17이었고 중간에 R18로 이동해도 이미 pinned R17 snapshot만 사용해야 한다. 각 shard fetch가 branch head를 다시 resolve하면 mixed revisions가 생긴다. 최초 resolve result를 transaction identity로 전달한다.

local directory에서 filename glob만 쓰는 path는 manifest가 없을 수 있다. 이 경우 index, shard count naming, header tensors, optional checksums를 이용해 self-consistent set을 만들고 결과 hash를 기록한다. 재현성을 위해 deployment artifact manifest를 별도로 생성하는 편이 낫다.

partial load rollback는 destination parameter object를 in-place mutate했는지 staging object를 썼는지에 따라 다르다. active model parameter를 직접 덮는 hot reload는 rollback가 사실상 불가능하거나 expensive하다. 별도 model instance/staging generation을 만들고 registry swap하는 이유다.

메모리 여유가 부족해 double-model staging가 어렵다면 service drain, replica-by-replica replacement, process restart 같은 deployment transaction을 쓴다. active bytes를 조각별 in-place 교체해 mixed generation을 노출하지 않는다. capacity 제약이 correctness boundary를 완화하지 않는다.

rank barrier가 timeout되면 늦은 rank의 prepare callback가 이미 abort된 L49를 commit하지 못해야 한다. transaction generation/status를 확인한다. retry L50 준비와 late L49 acknowledgment가 교차하는 fixture를 둔다. coordinator는 current transaction만 상태를 바꾼다.

model-ready probe는 parameter coverage만으로 끝나지 않는다. 작은 deterministic input의 logits/reference를 ranks/paths별로 비교해 name/order/transpose 오류를 잡는다. quantized models는 appropriate tolerance와 exact sentinel layers를 함께 쓴다. random response 품질 평가는 load identity 검증을 대신하지 않는다.

observability에는 tensor별 고 cardinality labels를 상시 노출하지 않는다. aggregate progress와 failure reason은 metrics, sampled/failing tensor coordinate는 trace/log에 둔다. load transaction ID로 manifest, rank, range, parameter edges를 join한다.

security 측면에서도 bounds와 size 검사는 중요하다. malformed header가 과도한 allocation나 out-of-bounds mapping을 유도하지 않게 parse limits를 둔다. 이 책의 목적은 security audit 전체가 아니지만 untrusted artifact를 정상 checkpoint처럼 가정하지 않는다.

운영 checklist는 pinned identity, exact bytes, parser bounds, logical shape/type, destination mapping, completion, coverage, atomic publish, cleanup 순서다. 어느 하나라도 “아마 framework가 처리”로 남으면 source reference나 fault fixture를 추가한다. 친절한 tutorial은 이 빈칸을 독자가 스스로 발견하게 한다.

reference 갱신 때는 line anchors가 이동했는지 검증하고 commit pin을 유지한다. behavior가 바뀌면 tutorial 계산의 constants나 branch 설명을 갱신한다. 원칙을 최신 구현 사실과 혼동하지 않으면서 둘 사이 evidence link를 유지한다.

최종 fault campaign은 load transaction의 commit 위치마다 한 번씩 실패를 넣는다. revision resolve 전 실패는 staging state가 없어야 한다. index resolve 뒤 실패는 manifest object만 정리한다. mmap/range open 뒤 실패는 handles와 temp buffers를 닫는다. parameter copy 뒤 실패는 device completion을 기다린 뒤 staging storage를 해제한다. all-rank prepare 뒤 commit 전 실패는 active registry16을 유지한다.

commit 직후 router publish 전 실패는 generation17 replica가 내부 ready지만 traffic에는 보이지 않는 상태일 수 있다. coordinator recovery가 publish를 재개할지 replica를 폐기할지 durable decision record로 정한다. router publish 뒤 실패는 generation17을 active로 인정하고 generation16 drain을 계속한다. commit 여부가 불명확한 채 양쪽을 free하지 않는다.

각 fault 위치의 expected metrics도 다르다. short range는 downloaded bytes 일부와 verified bytes 0, loaded slice 0이다. rank1 copy 실패는 rank0 staging coverage가 있어도 published generation는 16이다. all-rank success는 coverage exact, staging→active handoff, generation17 ready를 보인다. 동일 `load_failed_total` 뒤 state를 구분한다.

GGUF split fault는 마지막 part missing, directory count mismatch, type trait unsupported, mmap failure, GPU offload copy failure를 나눈다. parser 단계 실패와 backend 단계 실패는 cleanup owners가 다르다. partial model context가 caller에게 반환되지 않아야 한다.

성능 실험은 correctness campaign 뒤에 한다. safetensors full/eager와 slice/range, prefetch depth1/2/4, GGUF mmap/direct, TP ranks별 read duplication을 비교한다. wall load time, peak RSS, page faults, bytes read, GPU staging peak를 기록한다. faster path라도 transaction invariants를 완화하지 않는다.

range read가 network bytes를 줄여도 작은 requests가 많아 latency를 늘릴 수 있다. adjacent tensor ranges를 safely coalesce하고 concurrency를 bound한다. expected absolute offsets와 object identity를 보존하는 한 performance policy를 바꿀 수 있다. correctness layer와 scheduling policy를 분리한다.

TP ranks가 동일 full shard를 각각 읽으면 remote/storage amplification가 생긴다. rank-local ranges, node-local cache, broadcast를 검토할 수 있지만 source→destination coverage와 completion owner가 달라진다. optimization마다 failure semantics를 새 원장으로 작성한다.

terminal report에는 load time만 아니라 rollback time과 old generation availability를 넣는다. failed R17 load 동안 R16 requests가 정상 처리됐는지, staging cleanup가 다음 retry capacity를 회복했는지 본다. 서비스 무중단과 artifact correctness를 함께 평가한다.

마지막 승인자는 manifest R17, shard2 hash, tensor `[320,384)`, rank1 range `[352,384)`, sample file364, destination mapping, copy event, rank barrier, registry generation17을 한 trace에서 잇는다. GGUF path도 directory/type trait/block/destination를 같은 방식으로 잇는다. 이 왕복이 되는 것이 49장 tutorial의 최종 산출물이다.

승인 뒤에도 reload generation을 관측한다. 새 요청은 17, 기존 drain 요청은 16을 일관되게 사용하고 한 request 안에서 rank generations가 섞이지 않아야 한다. generation16 owner count가 0이 된 뒤 old mappings와 backend buffers를 해제한다. 시간 제한만으로 강제 free하지 않는다.

retry와 rollback 이력은 다음 배포의 evidence가 된다. L49 short-read, L50 verified reload, commit17의 manifest와 fault 결과를 연결한다. 동일 cache blob에서 오류가 반복되면 network retry를 늘리는 대신 object/cache identity를 격리한다. 같은 filename 성공 기록으로 다른 hash를 신뢰하지 않는다.

tutorial 독자는 마지막으로 숫자를 가리고 다시 계산한다. header length에서 payload base를 얻고 tensor relative range를 absolute로 바꾸며 TP slice와 destination layout offset을 구한다. 답이 source parser/loader trace와 다르면 어느 좌표 convention이 다른지 설명한 뒤 reference 원장을 갱신한다.

이 최종 검산은 특정 framework 명령 사용법보다 오래 간다. loader option과 class가 바뀌어도 pinned artifact identity, exact byte ownership, logical/physical slice, completion, coverage, atomic publish라는 질문은 유지된다. 새 format도 이 transaction에 끼워 넣어 안전하게 비교할 수 있다.

## 49.11 적재 계약을 identity·coverage·device commit으로 판정한다

이 장의 source는 Transformers v5.15.1 commit `550d7b3834670483a4df436541272c055dc364bf`, vLLM v0.27.1 commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, SGLang v0.5.18 commit `71de97b264b04dcd514cf904003028aefe9775c8`, llama.cpp v0.2.0 commit `bb4caa7540188872173c44d161602d9271386413`에 고정했다. Safetensors byte format은 commit `6eb4dc9a28ebce297606e0f4836bbf28839cacef`의 [고정 specification](https://github.com/huggingface/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/README.md#L76-L110)을 따른다.

이 좌표는 앞의 revision→shard→byte range→destination 생애를 재현하기 위한 source note이며 파일명이 같다는 사실만으로 content identity를 주장하는 근거가 아니다.

처음에는 파일이 있느냐만 보았다. 이제 loader correctness가 훨씬 긴 사슬이라는 것을 알 수 있다. revision이 config/index/shard의 content identity를 정하고, index가 name별 owning file을 정하며, safetensors header가 dtype/shape와 data-buffer-relative byte range를 정한다. mapping은 source name을 destination slice로 바꾸고 placement는 dtype/device storage를 만든다. 마지막 coverage가 누락과 중복을 닫는다.

Safetensors의 단순함도 정확히 이해해야 한다. 첫 8 byte N, N-byte JSON, contiguous data buffer와 relative offsets는 header-only inspection과 lazy mapping을 가능하게 한다. 그러나 PyTorch alias graph를 일반적으로 보존하지 않고, mmap이 GPU copy와 RAM pressure를 없애지도 않는다. tied weight는 model rule로 복원한다.

GGUF는 다른 답을 낸다. typed model metadata, tensor directory, alignment-padded data section과 GGML quant blocks가 한 format 안에 있다. split count/no와 first-file anchor가 file set을 만든다. tensor absolute offset은 각 split의 data base와 relative offset으로 계산한다. direct IO는 alignment-expanded read와 staging을 추가한다.

option도 label이 아니라 effective state와 consumer로 읽는다. eager는 whole shard bytes를, mmap은 file-backed pages와 mapping lifetime을, prefetch는 page cache를, threads와 buffer는 in-flight shard 수를, EP filter는 tensor data read 전 expert skip을 바꾼다. dtype/device/meta는 materialization 시점과 peak 위치를 바꾼다.

장애에서는 처음 갈라지는 단계로 돌아간다. missing file은 index set과 snapshot에서, corrupt file은 header/directory와 offset bounds에서, mixed revision은 commit/hash manifest에서 찾는다. wrong shape는 stored→logical→local→physical shape 사슬에서, wrong value는 mapping edge와 duplicate origin에서 찾는다.

좋은 loading report는 “20 GiB를 읽었다”로 끝나지 않는다. 어느 revision의 어느 index가 어떤 shards를 선택했고, 각 source name이 어느 offset에서 어떤 transform을 거쳐 어느 destination slice를 채웠으며, 어떤 option이 source lifetime과 peak를 바꿨는지 설명한다. 그 설명의 빈칸이 다음 조사 지점이다.

weight loading은 model 실행 전의 준비 작업이지만 correctness의 바깥이 아니다. 한 byte를 잘못 고르면 뒤의 모든 layer가 정확하게 잘못된 계산을 한다. 그래서 file set, byte range, name mapping, storage ownership과 coverage를 하나의 transaction처럼 검증하고, 완전히 닫힌 model만 다음 단계에 넘겨야 한다.

**적재 실패 결정 트리.** index가 key를 못 찾으면 shard directory·revision을, key는 있고 byte range가 틀리면 header offset·alignment를, tensor checksum은 맞고 model output이 다르면 layout transpose·quant metadata·tie를 조사한다. CPU staging은 맞고 GPU commit 뒤만 다르면 async copy completion과 destination owner를 본다. `strict=False`로 missing key를 덮는 대신 load transaction의 최초 불일치 단계에서 배포를 중단한다.

## 49.12 Reference — Safetensors·GGUF format fields와 byte directory

**Safetensors header·offset field reference.**

**처음 8 byte와 N-byte JSON**

Safetensors file의 처음 8 byte는 little-endian unsigned 64-bit header length `N`이다. 다음 N byte는 UTF-8 JSON header다. JSON은 `{`로 시작하고 trailing space padding이 가능하다. 그 뒤가 data buffer다.

예를 들어 첫 8 byte가 hex `80 00 00 00 00 00 00 00`이면 `N=128`이다. header는 file offset `[8,136)`이고 data buffer base는 absolute offset 136이다. header tensor entry의 `data_offsets`는 이 base에 상대적이다.

`x` entry가 dtype F16, shape `[2,3]`, offsets `[0,12]`라면 element 6개×2 byte=12 byte와 맞는다. 실제 file byte는 `[136,148)`다. `y`가 F32 `[2]`, offsets `[12,20]`이면 actual `[148,156)`다. end는 one-past offset이므로 size는 `END-BEGIN`이다.

이 산술이 parser validation의 핵심이다. `N`이 file size보다 크면 truncated/corrupt header다. JSON parse에 실패하거나 dtype이 unknown이면 header error다. shape product×dtype bytes와 range length가 다르면 tensor description이 inconsistent하다. END가 data buffer size를 넘으면 truncated data다.

**relative offset을 absolute로 착각하면 생기는 일**

entry `[0,12]`를 file offset으로 바로 읽으면 file magic 대신 8-byte N과 JSON 일부를 tensor로 해석한다. 값은 random처럼 보이고 shape cast가 가능하면 즉시 오류가 안 날 수도 있다. 항상 `8+N` data base를 더한다.

HTTP range metadata parser도 같은 규칙을 쓴다. 처음 bytes 0–7만 받아 N을 읽고, 8부터 `8+N-1`까지 header를 가져온다. 특정 tensor data를 range fetch할 때는 header relative offsets에 base를 더한다. shard index의 file mapping과 safetensors header offsets가 서로 다른 두 단계다.

offset은 tensor element index가 아니라 byte다. BF16/F16/F32에 따라 size가 다르다. packed quant tensor가 U8로 저장돼도 logical quant values 수와 U8 tensor shape가 conversion metadata에 따라 다를 수 있다. header는 저장 tensor의 dtype/shape를 말하며 dequantized logical shape를 자동 표현하지 않는다.

**overlap, hole, alias**

Safetensors는 arbitrary graph serialization이 아니라 tensor byte ranges를 담는다. parser는 offset bounds와 정렬된 data layout을 검증하고 overlapping ranges를 정상 alias 표현으로 취급하지 않는다. 두 state-dict names가 같은 PyTorch storage를 공유한다는 relation이 file range overlap으로 보존된다고 기대하면 안 된다.

tied embedding과 lm_head를 생각하자. model memory에서는 두 parameter names가 같은 storage를 가리킬 수 있다. save helper는 duplicate storage 때문에 한 canonical tensor만 직렬화하고 metadata/model tie rules로 관계를 복원할 수 있다. load report에서 빠진 alias name을 실제 missing independent weight와 구분해야 한다.

alias를 잘못 처리하는 반대 위험도 있다. checkpoint에 두 distinct tensors가 같은 값이라 해서 model이 tie해야 하는 것은 아니다. name rule과 model architecture가 storage sharing을 결정한다. value equality는 alias evidence가 아니다.

Safetensors data buffer가 holes 없이 이어진다는 format validation과 file-level alignment도 GGUF와 다르다. GGUF는 tensor마다 configured alignment padding을 둘 수 있다. 두 format의 offset 산술을 섞지 않는다.

**header-only inspection**

tensor dtype와 shape만 필요할 때 data를 전부 읽을 필요가 없다. SGLang current source의 [`weight_utils.py` 89–126행](https://github.com/sgl-project/sglang/blob/71de97b264b04dcd514cf904003028aefe9775c8/python/sglang/srt/model_loader/weight_utils.py#L89-L126)은 index에서 matching key의 shard를 찾고 file 앞에서 header length와 JSON만 읽어 dtype string을 찾는다.

이 path는 model config의 quant method를 보완하거나 checkpoint dtype을 검사할 수 있다. 하지만 첫 matching key의 dtype이 model 전체 dtype을 대표한다고 단정하면 안 된다. scale은 FP16, qweight는 U32/U8, norm은 BF16처럼 mixed storage일 수 있다. key pattern과 기대 field를 명시한다.

index가 없으면 첫 safetensors shard를 볼 수 있지만 target key가 그 file에 없을 수 있다. source가 no matching key에 무엇을 반환하는지 본다. header-only fast path도 index correctness에 의존한다.

**GGUF metadata·tensor directory field reference.**

**같은 “model file”이라는 말이 가리는 차이**

Safetensors shard를 읽던 습관으로 GGUF를 보면 `weight_map`을 찾게 된다. 그러나 GGUF는 model metadata key-values와 tensor directory를 file 안에 둔다. split model도 각 file의 GGUF metadata로 split count와 index를 확인한다. 외부 JSON index가 tensor name을 file에 배정하는 HF 방식과 다르다.

GGUF는 model architecture, tokenizer, quantization-related metadata를 함께 담을 수 있다. Safetensors header의 `__metadata__`는 string metadata일 뿐 model config 전체를 표준화한 directory가 아니다. Transformers model은 보통 별도 `config.json`과 tokenizer files를 사용한다.

두 format 모두 named tensors와 byte ranges를 갖지만 byte-size 계산도 다르다. Safetensors F16 `[K,N]`은 `KN×2`다. GGUF Q4_K 같은 type은 block struct 하나가 여러 logical values, scale/min sub-block metadata와 packed codes를 담는다. `KN×0.5`만으로 exact file bytes를 얻지 못한다.

**header, metadata KV, tensor directory**

llama.cpp current [`gguf.cpp` 560–793행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/ggml/src/gguf.cpp#L560-L793)은 file header와 metadata, tensor info를 parse한다. magic/version, tensor count, KV count가 먼저 온다. metadata entry는 key와 typed value를 가지며 array도 가능하다.

tensor directory entry는 tensor name, dimension 수와 extents, GGML type, data-section-relative offset을 가진다. parser는 dimension/product overflow와 type validity, duplicate tensor name, alignment을 확인해야 한다. directory는 tensor bytes 자체가 아니라 bytes를 해석할 schema다.

metadata의 `general.alignment`가 있으면 UINT32 type이어야 하고 값은 0이 아닌 power of two여야 한다. 없으면 default alignment를 쓴다. directory 뒤 file position을 alignment boundary로 pad한 곳이 data section base다.

Safetensors의 data base가 `8+N`으로 바로 정해지는 것과 다르다. GGUF는 variable-length KV와 tensor directory 전체를 읽은 뒤 `PAD(current_position,alignment)`를 계산한다. tensor absolute file offset은 `data_base + tensor_info.offset`이다.

**alignment fixture**

directory parsing이 끝난 file position이 1,013 byte이고 alignment가 32라고 하자. 다음 32-byte boundary는 1,024이므로 11 byte padding 뒤 data section이 시작한다. tensor A offset이 0이면 absolute 1,024다.

tensor A의 actual `ggml_nbytes`가 1,000 byte이면 다음 tensor를 단순 2,024에 두지 않는다. padded size는 `PAD(1000,32)=1024`, tensor B expected relative offset은 1,024, absolute는 2,048이다. A 뒤 24 byte alignment padding이 있다.

tensor directory가 B offset을 1,000이라고 기록했다면 parser가 expected 1,024와 다르다고 fail해야 한다. overlap이 없어 보이더라도 format의 aligned layout contract를 위반한다. direct IO는 더 큰 device alignment를 요구해 read range를 추가 확장할 수 있지만 tensor logical offset은 GGUF alignment를 따른다.

alignment overhead는 tensor 수와 size 분포에 달렸다. 큰 matrices에서는 작지만 수천 small tensors에서는 누적될 수 있다. file size를 tensor logical bytes 합과 바로 비교하지 않고 header/directory와 per-tensor padding을 더한다.

**quant block fixture**

GGML quant type은 block size `QK` values와 block struct byte size `type_size`를 가진다. logical elements가 `ne`이면 storage block count가 보통 `ne/QK` 제약을 따르고 bytes는 `blocks×type_size`다. exact traits는 current GGML type table을 사용한다.

설명용으로 한 block이 32 values를 18 byte에 담는 Q4-like format을 가정하자. 1,024 values는 32 blocks, bytes는 576이다. naive `1024×4/8=512`보다 64 byte 크다. block마다 scale metadata 2 byte가 있기 때문이다. 실제 Q4_0 struct의 field와 size를 source에서 고정한다.

K-quant super-block은 scales/mins와 codes를 더 복잡하게 pack한다. element 일부만 TP slice하려면 block boundary가 중요하다. arbitrary logical element offset에서 raw bytes를 자르면 block metadata를 잃는다. GGUF loader가 GPU backend로 tensor를 offload할 때 packed block storage를 보존하고 quantized matmul kernel이 소비할 수 있다.

따라서 “GGUF는 load하면서 FP16으로 변환된다”는 보편 문장은 틀리다. backend와 tensor type에 따라 packed GGML tensor가 CPU/GPU buffer에 그대로 materialize된다. unsupported backend op에서 dequant temporary가 생길 수 있지만 별도 execution path다.

**split filename과 metadata**

llama.cpp의 [`llama_get_list_splits`와 loader 80–105, 532–668행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L80-L105)은 `name-00002-of-00004.gguf` 같은 path에서 prefix와 expected split paths를 만든다. loader main path는 [`532–668행`](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L532-L668)에서 first file metadata를 읽는다.

split count가 1보다 크면 first file의 split index가 0이어야 한다. 사용자가 두 번째 split을 entry로 주면 나머지 files가 있어도 오류다. first file이 global metadata와 complete split discovery의 anchor다.

expected split count와 supplied custom list length가 같아야 한다. 각 subsidiary file을 열어 split index key가 expected loop index와 같은지 확인한다. filename만 00002이고 metadata는 split 3이라면 fail한다. file content identity를 basename보다 신뢰한다.

각 split은 별도 GGUF context, tensor directory와 data base offset을 가진다. main file의 data base를 subsidiary tensor에 사용하면 wrong byte를 읽는다. source comment가 subsidiary file에 main metadata tensor offset을 쓰지 말라고 강조하는 이유다.

tensor names는 all files의 inventory로 합쳐지고 owning file index와 offset을 기록한다. duplicate name이 다른 split에 있으면 split tensor 조각인지 illegal duplicate인지 format/model loader contract를 확인한다. 일반 GGUF split은 HF weight_map처럼 한 tensor name을 여러 arbitrary byte chunks로 나누는 것과 다를 수 있다.

**mmap path와 direct-read path**

GGUF loader는 file mapping을 backend tensor buffer로 사용할 수 있고, option과 device placement에 따라 chunked read와 copy를 사용할 수 있다. mmap이면 tensor data pointer는 owning file mapping의 `data_base+offset`을 가리킬 수 있다. mapping은 model tensor가 살아 있는 동안 유지돼야 한다.

direct IO 또는 aligned read path는 file/device가 요구하는 alignment를 계산한다. current [`llama-model-loader.cpp` 1447–1635행](https://github.com/ggml-org/llama.cpp/blob/bb4caa7540188872173c44d161602d9271386413/src/llama-model-loader.cpp#L1447-L1635)은 maximum read alignment와 staging buffer 크기를 정하고 tensor offset을 aligned down, end를 aligned up한다.

tensor actual range가 `[offset,offset+n_size)`이고 file read alignment가 4096이라 하자. offset 10,000은 aligned down 8,192, leading padding 1,808이다. end 16,000은 aligned up 16,384다. physical read range는 8,192 byte지만 actual tensor bytes는 그 안의 6,000 byte다.

첫 chunk에서는 leading padding을 건너뛰고 마지막 chunk에서는 trailing padding을 자른다. destination에는 exactly tensor bytes만 copy한다. alignment-expanded IO byte와 logical tensor byte를 구분한다. small unaligned tensors가 많으면 read amplification이 커질 수 있다.

double host buffers와 asynchronous device copy를 쓰면 read와 transfer가 겹칠 수 있지만 staging peak는 buffer count×buffer size다. mmap path와 direct-read path의 peak/IO 모델을 같은 “GGUF load” 숫자로 합치지 않는다.
