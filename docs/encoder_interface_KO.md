# P08 공개 인터페이스

고정327 encoder와 train-only supervisor의 연구용 인터페이스다. 정식 자료·checkpoint·대화decoder·공유기억은 포함하지 않는다.

## 어휘와 입력

P02 encode_text('가')는 jamo_ids=(4353,4450)이다. 별도 to_fixed_tokens만 이를 (1,20)으로 변환하며 encoded 원문 객체와 reconstruction_map을 유지한다.
고정 ID는 PAD0 / 현대자모1..67 / byte68..323 / marker324,325 / 예약UNK326. UNK는 변환에서 생성하지 않는다. 문장 의미·정답을 파싱하지 않는다.
원문 상한512bytes, 최대 확장1536tokens. ASCII512는 START+byte+END ×512로 확장된다. normalize/truncate를 하지 않는다.
make_fixed_batch는 EncodedText tuple과 호출자 item_ids를 받아 별도 FixedTokenBatch를 반환한다. to_tensor_batch는 기존 P02 EncodedBatch를 거절한다. TensorBatch는 CPU int64 IDs/lengths, bool valid_mask이고 모델이 shape/PAD/prefix/range/port를 재검증한다.

## 추론 연결 예시

```python
from idea_model.codec import encode_text
from idea_model.contracts import new_state, RunMode
from p08_encoder import TextRelationModel, make_fixed_batch, to_tensor_batch

owner, episode, seed = "individual-a", "episode-a", 31101
model = TextRelationModel(owner, seed=seed)
batch = to_tensor_batch(make_fixed_batch((encode_text("각A"),), item_ids=("observation-a",)))
state = new_state(owner, episode, seed=seed)
prediction = model(batch, (state,), owner_id=owner, episode_ids=(episode,),
                   seed=seed, mode=RunMode.TEST)
```

relation_logits는 float32[B,4,4], 순서box/tool/from/to이고 allowed_logits는 float32[B,2]다. allowed의 학습 class는0=false/1=true이며 이것은 예측이다. 정답입력·생성기solve/render와 연결하지 않는다.
family=idea-relational-gru-v1, port=text-relations-327-v1. native77/B/이진 코어와 다른 포트이며 원본 동등/언어학습 성공을 주장하지 않는다.
observed_mask=False인 행은 빈 관측이다. logits0은 유효 행동/관계 예측으로 사용하지 않고 downstream에서 관측을 제외한다. 해당 행은 입력state 객체 그대로 반환한다. 빈batch는0행 output/빈state tuple이다.
추론state hidden은64float tuple이며 memoryview는 그대로 보존한다. active행 trace는 deepcopy하므로 입력trace/다른episode로 mutable값이 퍼지지 않는다. 공유기억/기억쓰기/decoder/action환경은 이 부품에 없다.
모드가selection/test이면 no_grad이며 가중치·기존grad는 갱신하지 않는다. 계산은CPU float32로 고정하여 외부 기본장치/autocast 설정을 따르지 않는다.

## train-only 연결

```python
from p08_encoder import new_train_state, TrainTargets, TrainOnlySupervisor
# labels는 실제 자료를 읽지 않은 손작성 tiny fixture의 예시이며 의미 정답을 계산하지 않는다.
# relation labels: CPU int64[B,4], 각 class0..3
# allowed labels: CPU int64[B], class0..1
trainer = TrainOnlySupervisor(model, owner_id=owner, seed=seed, mode=RunMode.TRAIN, lr=0.001)
train_state = new_train_state(owner, episode, seed=seed)
train_prediction = model(batch, (train_state,), owner_id=owner, episode_ids=(episode,),
                         seed=seed, mode=RunMode.TRAIN)
# targets = TrainTargets(batch.item_ids, (episode,), relation_labels, allowed_labels)
# loss = trainer.relation_loss(train_prediction, targets)
```

학습state는 별도 TrainEpisodeState다. 기존 CoreState의 Tensor거절 규약에 자동변환/예외를 추가하지 않는다. 이전 new_states를 같은model/owner/episode/seed의 다음공개관측에 전달하여 전체episode BPTT를 유지한다. 빈관측은 그래프를 끊지 않는다. 다른모델 graph, detached graph, train/eval 상태혼용을 거절한다.
Trainer는 일치하는train prediction과TrainTargets만 받아 관계loss를 계산한다. 일반 TrainingTarget/TestTarget 본문은 해석하지 않는다. eval capability와 eval prediction의label 소비는 거절한다. item/episode/shape/dtype/range가 틀린labels는loss 전에 거절하고weights/state/grads를 변경하지 않는다.
관계loss 분모는 유효episode×5head이며 PAD/빈행은 제외한다. 미구현 조건일관성.25/outcome.5/표현.25를 관계loss에 재분배하지 않는다.
optimizer는 각모델의별도Adam이며lr는호출자가명시한다. 전체episode graph를만드는동안step하지않는다. 마지막loss/backward 뒤 허용된step을하고 reset_train_state로graph를정리한다. 정식학습runner는현재구현범위가아니다.
모든관측이빈episode/batch에서는caller가optimizer.step을건너뛴다. 빈loss0·gradient0은검증하지만 기존Adam momentum이있는상태에서빈step까지가중치불변임을보장하는update-wrapper는없다. 이협조적인capability경계는동일OS사용자의직접Tensor/파일변조를막는보안벽이아니다.

## 검증·비용 범위

새초기화 embedding10464+GRU18816+관계/allowed1170=30450parameters, float32weights121800bytes. gradient/optimizer/activation/프레임워크RAM은이수치에포함되지않으며 보호도구Jobpeak에별도기록된다. outcome195parameters는현재부품범위밖이다.
독립fixture는정상BPTT/PAD/빈입력/owner/episode/seed/reset/weights·optimizer격리/실패원자성/float32계약을검사한다. tiny one-step을모델이의미를배웠다는성과로보고하지않는다.
P05봉인test/정식자료/실사용자자료/원본native/B/이진코어를읽거나평가하지않았다. 통합회귀와정식학습은부모의다음단계다.
